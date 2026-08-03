"""ComfyUI custom nodes for LLM-powered prompt enhancement via llama-server.

Supports KREA-2 Text-to-Image (krea2-t2i), LTX 2.3 10Eros Image-to-Video (ltx2.3-10eros-i2v),
and MiniMax H3 Reference-to-Video (minimax-h3-r2v).

Each preset handles both SFW and NSFW content via in-prompt directives.
The target model is determined automatically from the selected preset.
"""

from __future__ import annotations

from pathlib import Path

from comfy_api.latest import io

ComfyNode = io.ComfyNode

# Support both package imports (ComfyUI) and direct imports (tests)
try:
    from .llm_client import enhance_prompt
    from .presets import (
        get_default_preset,
        load_preset,
        list_presets,
    )
except ImportError:
    from llm_client import enhance_prompt
    from presets import (
        get_default_preset,
        load_preset,
        list_presets,
    )

# Default server path - works if llama-server is in PATH
DEFAULT_SERVER_PATH = "llama-server"

# Default model: Gemma 4 31B NVFP4 Turbo (~18 GB)
DEFAULT_MODEL_PATH = (
    "C:/Users/maxkr/.lmstudio/models/lmstudio-community/"
    "gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-NVFP4-turbo-NVFP4.gguf"
)

def _get_comfy_base_path() -> str:
    """Get the ComfyUI base directory for relative model paths."""
    custom_nodes_dir = Path(__file__).resolve().parent.parent
    return str(custom_nodes_dir.parent)


def _resolve_path(raw: str, fallback: str = "") -> str:
    """Resolve a path string: relative -> absolute via ComfyUI base.

    Bare command names (no path separators) are returned as-is so
    the OS can resolve them via PATH - this is how llama-server
    should be found when it's installed globally."""
    import os

    path = raw.strip()
    if not path:
        return fallback
    # If it contains a separator it's a file path; otherwise treat as a
    # command name and let the OS resolve it via PATH.
    if not os.path.isabs(path) and ("/" in path or "\\" in path):
        comfy_base = _get_comfy_base_path()
        path = os.path.join(comfy_base, path)
    return path


def _build_preset_options():
    """Build (display_name, key) pairs for the preset dropdown.

    Returns a list of display names and a dict mapping display -> key.
    The dropdown stores the display name; we resolve to the internal key at runtime.
    """
    presets = list_presets()
    if not presets:
        return ["no presets found"], {}

    display_to_key = {}
    display_names = []
    for p in presets:
        display_names.append(p.display_name)
        display_to_key[p.display_name] = p.key

    return display_names, display_to_key


def _get_default_preset_display(display_to_key: dict) -> str:
    """Get the display name of the default preset."""
    default = get_default_preset()
    if default:
        return default.display_name
    return ""


# --- Shared state for preset options (built once) ---
_preset_display_names: list[str] | None = None
_display_to_key: dict[str, str] | None = None


def _ensure_preset_options():
    global _preset_display_names, _display_to_key
    if _preset_display_names is None:
        _preset_display_names, _display_to_key = _build_preset_options()


def _resolve_preset_key(preset_display: str) -> str | None:
    """Resolve a preset display name to its internal key. Returns system prompt or None."""
    _ensure_preset_options()

    preset_key = preset_display
    if _display_to_key and preset_display in _display_to_key:
        preset_key = _display_to_key[preset_display]

    system_prompt = load_preset(preset_key)
    if not system_prompt:
        system_prompt = load_preset(preset_display)
    if not system_prompt:
        default = get_default_preset()
        if default:
            system_prompt = load_preset(default.key)
    return system_prompt


class PromptEnhancer(ComfyNode):
    """Enhance a prompt using a local LLM (llama-server).

    Supports KREA-2 T2I and LTX 2.3 10Eros I2V via presets.
    Each preset handles both SFW and NSFW content automatically.
    The target model is determined by the selected preset.

    This node:
    1. Unloads all ComfyUI models to free VRAM
    2. Spawns llama-server with your GGUF model
    3. Sends your prompt + system preset to the LLM
    4. Retries until a quality prompt is returned
    5. Kills the server when done

    Each execution produces a unique, enhanced prompt.
    When bypassed, the original prompt passes through unchanged.
    """

    @classmethod
    def define_schema(cls):
        _ensure_preset_options()
        display_names = _preset_display_names or ["no presets found"]
        default_display = _get_default_preset_display(_display_to_key or {})

        return io.Schema(
            node_id="PromptEnhancer",
            display_name="Prompt Enhancer",
            category="Prompt Enhancer",
            description=(
                "Enhance prompts using a local LLM via llama-server. "
                "Supports KREA-2 (T2I) and LTX 2.3 10Eros (I2V) via presets. "
                "Each preset handles both SFW and NSFW content automatically. "
                "Unloads models, runs LLM with retry loop, returns unique enhanced prompt."
            ),
            inputs=[
                # Prompt first so it passes through on bypass
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    placeholder=(
                        "Describe the scene you want to generate...\n\n"
                        "Example: A woman walking through a rainy street at night\n\n"
                        "You can also include extra context like:\n"
                        "Here is additional info from the used loras: character has pink hair, blue eyes"
                    ),
                ),
                io.Combo.Input(
                    "preset",
                    options=display_names,
                    default=default_display,
                    tooltip=(
                        "System prompt preset that guides how the LLM enhances your prompt.\n"
                        "Each preset handles both SFW and NSFW content automatically.\n"
                        "The target model is determined by the preset."
                    ),
                ),
                io.String.Input(
                    "llm_model_path",
                    default=DEFAULT_MODEL_PATH,
                    placeholder=(
                        "Full path to your .gguf model file\n"
                        "Example: C:/models/qwen2.5-7b-instruct-q4_k_m.gguf"
                    ),
                    tooltip="Path to the GGUF model file for llama-server",
                ),
                io.String.Input(
                    "llama_server_path",
                    default=DEFAULT_SERVER_PATH,
                    placeholder="llama-server or full path to llama-server binary",
                    tooltip="Path to the llama-server executable (or 'llama-server' if in PATH)",
                ),
                io.Int.Input(
                    "ctx_size",
                    default=16000,
                    min=2048,
                    max=131072,
                    step=1024,
                    tooltip="Prompt context window size in tokens. 16000 gives the LLM enough headroom for system prompt, user prompt, and generation.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**63 - 1,
                    step=1,
                    tooltip="Random seed for LLM generation. Use -1 for random seed each time.",
                    control_after_generate=io.ControlAfterGenerate.randomize,
                ),
                # Optional inputs
                io.String.Input(
                    "mmproj_path",
                    optional=True,
                    default="",
                    placeholder="Path to mmproj-*.gguf multimodal projector file",
                    tooltip="Multimodal projector GGUF for vision input. Required when using reference images with a multimodal model.",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip=(
                                "Reference image sent to the LLM alongside your prompt. "
                                "Useful for I2V presets: the LLM sees the image(s) and tailors the prompt to match."
                            ),
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                    tooltip="Dynamic reference images (0-9). The LLM sees all connected images.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "enhanced_prompt",
                    display_name="enhanced_prompt",
                    tooltip="The LLM-enhanced prompt text. Passes through original prompt when node is bypassed.",
                ),
            ],
            hidden=[
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        preset: str,
        llm_model_path: str,
        llama_server_path: str,
        ctx_size: int,
        seed: int,
        ref_images: dict[str, any] | None = None,
        mmproj_path: str = "",
        **kwargs,
    ) -> io.NodeOutput:
        if not prompt or not prompt.strip():
            return io.NodeOutput("")

        # Load preset system prompt
        system_prompt = _resolve_preset_key(preset)
        if not system_prompt:
            return io.NodeOutput(prompt)

        user_prompt = prompt.strip()

        # Resolve paths
        model_path = _resolve_path(llm_model_path)
        server_path = _resolve_path(llama_server_path, DEFAULT_SERVER_PATH)
        mmproj = _resolve_path(mmproj_path) if mmproj_path else ""

        # Collect reference images from autogrow dict
        images = []
        if ref_images:
            for name in sorted(ref_images.keys()):
                img = ref_images[name]
                if img is not None:
                    images.append(img)

        # Call the LLM client
        result = enhance_prompt(
            server_path=server_path,
            model_path=model_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            ctx_size=ctx_size,
            seed=seed,
            images=images,
            mmproj_path=mmproj,
        )

        return io.NodeOutput(result if result else prompt)


class PromptEnhancerBatch(ComfyNode):
    """Enhance multiple prompts using a local LLM.

    Each prompt gets its own server session with quality validation and retry loop.
    """

    @classmethod
    def define_schema(cls):
        _ensure_preset_options()
        display_names = _preset_display_names or ["no presets found"]
        default_display = _get_default_preset_display(_display_to_key or {})

        return io.Schema(
            node_id="PromptEnhancerBatch",
            display_name="Prompt Enhancer (Batch)",
            category="Prompt Enhancer",
            description=(
                "Enhance multiple prompts (one per line) with retry loop. "
                "Each preset handles both SFW and NSFW content automatically. "
                "Each prompt gets its own server session with quality validation."
            ),
            inputs=[
                io.String.Input(
                    "prompts",
                    multiline=True,
                    dynamic_prompts=True,
                    default="",
                    placeholder=(
                        "Enter one prompt per line:\n\n"
                        "Prompt 1\nPrompt 2\nPrompt 3"
                    ),
                ),
                io.Combo.Input(
                    "preset",
                    options=display_names,
                    default=default_display,
                    tooltip=(
                        "System prompt preset. Each preset handles both SFW and NSFW content automatically. "
                        "Target model determined by preset."
                    ),
                ),
                io.String.Input(
                    "llm_model_path",
                    default=DEFAULT_MODEL_PATH,
                    placeholder="Full path to your .gguf model file",
                ),
                io.String.Input(
                    "llama_server_path",
                    default=DEFAULT_SERVER_PATH,
                ),
                io.Int.Input(
                    "ctx_size",
                    default=16000,
                    min=2048,
                    max=131072,
                    step=1024,
                    tooltip="Prompt context window size in tokens.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**63 - 1,
                    step=1,
                    control_after_generate=io.ControlAfterGenerate.randomize,
                ),
                io.String.Input(
                    "mmproj_path",
                    optional=True,
                    default="",
                    placeholder="Path to mmproj-*.gguf multimodal projector file",
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_image",
                            tooltip="Optional reference image sent to the LLM alongside each prompt.",
                        ),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    "enhanced_prompts",
                    display_name="enhanced_prompts",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompts: str,
        preset: str,
        llm_model_path: str,
        llama_server_path: str,
        ctx_size: int,
        seed: int,
        ref_images: dict[str, any] | None = None,
        mmproj_path: str = "",
        **kwargs,
    ) -> io.NodeOutput:
        # Parse prompts (one per line)
        raw_prompts = [p.strip() for p in prompts.split("\n") if p.strip()]
        if not raw_prompts:
            return io.NodeOutput("")

        # Load preset
        system_prompt = _resolve_preset_key(preset)
        if not system_prompt:
            return io.NodeOutput("\n".join(raw_prompts))

        # Resolve paths
        model_path = _resolve_path(llm_model_path)
        server_path = _resolve_path(llama_server_path, DEFAULT_SERVER_PATH)
        mmproj = _resolve_path(mmproj_path) if mmproj_path else ""

        # Collect reference images
        images = []
        if ref_images:
            for name in sorted(ref_images.keys()):
                img = ref_images[name]
                if img is not None:
                    images.append(img)

        # Enhance each prompt
        results = []
        for i, prompt in enumerate(raw_prompts):
            print(f"  [PromptEnhancer Batch] Enhancing prompt {i+1}/{len(raw_prompts)}...")

            result = enhance_prompt(
                server_path=server_path,
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=prompt,
                ctx_size=ctx_size,
                seed=seed,
                images=images,
                mmproj_path=mmproj,
            )
            results.append(result if result else prompt)

        return io.NodeOutput("\n".join(results))


# Node registration
NODE_CLASS_MAPPINGS = {
    "PromptEnhancer": PromptEnhancer,
    "PromptEnhancerBatch": PromptEnhancerBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEnhancer": "Prompt Enhancer",
    "PromptEnhancerBatch": "Prompt Enhancer (Batch)",
}
