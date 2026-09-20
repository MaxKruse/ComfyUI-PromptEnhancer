# Repository Guidelines

## Project Overview

ComfyUI custom node that rewrites image/video generation prompts using a **local LLM**. Each execution spawns a temporary `llama-server` (llama.cpp) with a GGUF model, sends the user prompt plus a per-target-model system-prompt preset via the OpenAI-compatible API, retries until a quality gate passes, kills the server, and returns the enhanced prompt (original prompt on any failure or bypass). No API keys, no network - fully local.

Drop-in `custom_nodes/` package: **not** an installable Python package (no `pyproject.toml`/`setup.py`). Single registered node: `PromptEnhancer` (display name "Prompt Enhancer").

## Architecture & Data Flow

Flat module layout - 3 source modules + entry point:

```
__init__.py   -> re-exports NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS (try/except ImportError -> {})
nodes.py      -> PromptEnhancer node (comfy_api.latest.io schema), preset dropdown + path helpers
llm_client.py -> llama-server lifecycle, HTTP client, quality gate, enhance_prompt() public entry
presets.py    -> presets/*.txt discovery, PresetInfo metadata, target-model prefix mapping
```

Per-execution data flow (`PromptEnhancer.execute()` -> `enhance_prompt()` in `llm_client.py`):

1. `_resolve_preset_key()` maps the Combo display name -> preset file key -> `load_preset()` text (falls back: raw display name -> first preset alphabetically).
2. `free_gpu_memory()` - `comfy.model_management.unload_all_models()`, CUDA cache + `gc.collect()` (llama-server gets the VRAM).
3. `find_free_port()` (random ephemeral 49152-65535) -> `Popen` llama-server (`CREATE_NO_WINDOW` on Windows; daemon thread drains stdout into an 80-line deque for crash diagnostics) -> poll `GET /health` every 0.5 s (<=120 s) -> `GET /v1/models` for model id.
4. `UNCENSORED_PREFIX` is **silently prepended** to the preset system prompt - the preset file is not the full system message.
5. Retry loop (`max_retries`, default 5): `POST /v1/chat/completions` (urllib, `Bearer no-key`, **seed per request in the payload** - attempt N sends `(base_seed + N - 1) % 2**32`, so retries vary; temperature/top-p/top-k are NOT in the payload, server-side via `extra_flags` only), each request in an interruptible worker thread. Accept first result passing `is_good_prompt()` (word floor >= `min_words`, refusal-phrase blacklist, Jaccard word overlap > 0.85 = echo); otherwise keep the longest **clean** candidate - refusals are never returned (all-refusal -> `None` -> original prompt). Each attempt prints inference speed (prompt/generated tok/s) and ctx tokens used from the response `timings`; a context-overflow HTTP error prints an `OUT OF CONTEXT` hint. The seed is no longer passed as a CLI `--seed` to llama-server (per-request payload seed overrides any server default).
6. `kill_server()` in `finally` (terminate -> 5 s -> kill) -> return best result; node returns the **original prompt** when the result is falsy or empty.

Key architectural properties:

- **New `comfy_api.latest.io` schema only** - `@classmethod define_schema(cls) -> io.Schema`, `execute()` returns `io.NodeOutput`. No legacy `INPUT_TYPES`/`RETURN_TYPES`/`validate()` (the tests' `_get_input_types()` helper normalizes V1/V3 shapes).
- **No asyncio, no caching, no persistence** - every run is a cold spawn (full GGUF reload, up to 120 s startup).
- **Multimodal**: reference images (Autogrow slots `ref_image_0..8`, torch tensors `[H,W,3]` or `[N,H,W,3]`) are JPEG-encoded (PIL, q85) to base64 `image_url` parts. They are **silently dropped** (logged) unless `mmproj_path` is set.
- **Bypass-safe**: empty/whitespace prompt or disabled node passes `prompt` through unchanged (that is why `prompt` is the first input).

## Key Directories

| Path | Purpose |
|------|---------|
| `nodes.py` | The one node class + schema, node registration, path/preset helpers |
| `llm_client.py` | Entire llama-server lifecycle + HTTP client + quality gate (public entry: `enhance_prompt()`) |
| `presets.py` | Preset discovery API: `list_presets`, `load_preset`, `get_default_preset`, `get_preset_by_key`, `TARGET_MODEL_LABELS` |
| `presets/` | System-prompt `.txt` files - **the extension point**. Naming: `<target>-<name>.txt` (e.g. `krea2-t2i-portrait.txt`); target prefix drives the dropdown label. Bundled: `krea2-t2i`, `ltx2.3-10eros-i2v`, `ltx2.5-i2v`, `minimax-h3-base`, `minimax-h3-r2v`, `qwenimage2.1-t2i`, `qwenimage2.1-i2i` |
| `tests/` | One pytest module per source module (`test_nodes`, `test_llm_client`, `test_presets`) + `conftest.py` |

## Development Commands

There is no build step. The node is imported by ComfyUI from `custom_nodes/`.

```bash
# Install (README)
cd ComfyUI/custom_nodes
git clone https://github.com/MaxKruse/ComfyUI-PromptEnhancer.git

# Tests (run from the repo root; verified: 77 passed)
pytest
pytest tests/test_presets.py   # single module
```

No lint, formatter, type-check, or CI is configured anywhere (no ruff/flake8/mypy configs, no `.github/`). `requirements.txt` is comment-only: **zero Python dependencies** beyond ComfyUI core; the external `llama-server` binary (in PATH or via the `llama_server_path` input) and a GGUF file are the real prerequisites.

## Code Conventions & Common Patterns

- **Lazy heavy imports**: `torch`, `PIL`, `comfy.model_management`, `shlex`, `os` are imported **inside functions** so `llm_client.py`/`presets.py` import cleanly without ComfyUI (tests depend on this).
- **Dual import pattern**: `try: from .x import ... except ImportError: from x import ...` (ComfyUI package import vs. test direct import).
- **Error handling**: the client layer never raises domain errors - it logs via the module `logger` + `_print_safe()` (thread-safe, survives cp1252 consoles) and returns `None`; the node degrades to the original prompt. The only exception that propagates is ComfyUI's `InterruptProcessingException` (`_check_interrupt()`, no-op outside ComfyUI).
- **State**: one module-level lazy singleton in `nodes.py` (`_preset_display_names` / `_display_to_key` via `_ensure_preset_options()`). `llm_client.py` is stateless functions. No DI, no globals elsewhere.
- **Naming**: private helpers `_snake_case`, public functions snake-case verbs (`load_preset`, `enhance_prompt`), constants UPPER_SNAKE (`UNCENSORED_PREFIX`, `DEFAULT_MODEL_PATH`, `PRESETS_DIR`). `from __future__ import annotations` and `str | None` unions throughout.
- **Path rules**: `llm_model_path` and `mmproj_path` **must be absolute** (relative -> log error -> treated as empty -> original-prompt fallback); `llama_server_path` may be a bare PATH command (`/` or `\` decides). `_get_comfy_base_path()` in `nodes.py:54` is an unused leftover helper.
- **Machine-specific defaults**: `DEFAULT_MODEL_PATH` and `DEFAULT_EXTRA_SERVER_ARGS` in `nodes.py` hardcode the author's `C:/Users/maxkr/...` paths. They are just default input values (pinned by tests), not config loading.
- **Seed semantics**: `seed` is the base sampling seed, sent in the chat completion payload per attempt: attempt N sends `(base + N - 1) % 2**32`; negative `seed` = random 32-bit base per run. Not passed to llama-server as CLI `--seed` (removed).
- **README is the living doc** - update it when node inputs, presets, flags, or defaults change (see Testing & QA for what is pinned).

## Important Files

- `__init__.py` - ComfyUI entry point; empty-mapping fallback makes the package importable for tests.
- `nodes.py` - schema source of truth for all node inputs (`prompt`, `preset`, `llm_model_path`, `llama_server_path`, `ctx_size`, `seed`, `max_retries`, `min_words`, `mmproj_path`, `extra_server_args`, `ref_images`); output `enhanced_prompt`.
- `llm_client.py` - `enhance_prompt()` is the single public entry; everything else (`_start_llama_server`, `wait_for_server`, `chat_completion`, `is_good_prompt`, `kill_server`, ...) is private.
- `presets/*.txt` - prompt-engineering content; tests pin specific substrings in them (see below).
- `tests/conftest.py` - sys.path bootstrap: injects the ComfyUI root (4 parents up from `tests/`) and the package root. **The node must live at `<ComfyUI>/custom_nodes/ComfyUI-PromptEnhancer`** or tests break.
- `README.md` - the entire documentation surface (install, node inputs, presets table, extra-flags reference, troubleshooting).
- `pytest.ini` - `testpaths = tests`, `python_files = test_*.py`, `norecursedirs = ..`.

## Runtime/Tooling Preferences

- Python version is not declared - inherits whatever ComfyUI core runs on (tests have run under CPython 3.13).
- No package manager involved; no venv, no env vars, no `.env`, no lockfile. All configuration is via **node inputs** - model selection is never an env var.
- Windows-first (author's paths, `Thumbs.db` in `.gitignore`, `CREATE_NO_WINDOW`); tests that spawn real processes depend on Windows process-tree kill.
- `pytest` must come from the host environment (not declared in `requirements.txt`).

## Testing & QA

- Framework: pytest, plain test functions (no classes/fixtures), one module per source module. Run with `pytest` from the repo root. No coverage tooling, no CI.
- **Mocking pattern**: fake `comfy`/`comfy.model_management` `types.ModuleType` objects injected into `sys.modules` via monkeypatch (`_install_fake_comfy()` in `tests/test_llm_client.py`) - reuse this pattern when touching `comfy.*` interaction. Server-lifecycle tests use a **real** `subprocess.Popen` (recording wrapper) of a python stand-in process, plus `time.monotonic` timing asserts to catch regression to blocking waits.
- **Tests pin behavior, not structure**: defaults (`ctx_size` 20000, `min_words` 50, Muse-Glimmer GGUF default, DFlash `extra_server_args` flags), input ordering (`prompt` first for bypass), negative assertions (absent legacy inputs like `target_model`/`reference_image`, generic non-LTX names), and preset-file **content substrings** (NSFW directives, SNOFS LoRA vocabulary, the "photograph not photorealistic" rule, MiniMax section names, Qwen-Image 2.1 plain-text/no-JSON output and its `<Picture N>` tags). Changing a preset's wording or a node default breaks tests by design - update the test or restore the contract deliberately. The seed wiring is also pinned: `chat_completion` must send `seed` in the payload, the retry loop must increment it per attempt, and `build_command` must not emit a CLI `--seed`. Quality-gate pins: a faithful expansion that preserves every original word must pass, near-verbatim echoes must fail, refusals must be excluded from the retry fallback, each reference image part must be preceded by a `<Picture N>` label, and `timings`/context-overflow responses must print to the console.
- Preset content tests break on rewording of `presets/*.txt`; discovery-driven loops (`test_all_presets_load_without_error`, unique keys/display names) automatically cover new preset files.
- Timing-sensitive tests exist (`<1.5 s`/`<2.5 s` asserts on real process spawn/kill) - expect occasional slowness, not failure.
