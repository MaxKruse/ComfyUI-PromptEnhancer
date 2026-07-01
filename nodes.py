"""ComfyUI custom nodes for LLM-powered prompt enhancement via llama-server.

Supports multiple target models:
  - LTX 2.3 Text-to-Video (ltx-t2v)
  - LTX 2.3 Image-to-Video (ltx-i2v)
  - KREA-2 Text-to-Image (krea2-t2i)

The target model is determined automatically from the selected preset.
"""

from __future__ import annotations

import os
from pathlib import Path

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

# Default model: Gemma 4 12B QAT Q4_0 (~6.6 GB)
DEFAULT_MODEL_PATH = (
    "C:/Users/maxkr/.lmstudio/models/lmstudio-community/"
    "gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf"
)

# Default llama-server flags matching ~/.llama-cpp.ini
DEFAULT_EXTRA_FLAGS = "--no-mmap --threads 12 -ngl 99"


def _get_comfy_base_path() -> str:
    """Get the ComfyUI base directory for relative model paths."""
    custom_nodes_dir = Path(__file__).resolve().parent.parent
    return str(custom_nodes_dir.parent)


def _resolve_path(raw: str, fallback: str = "") -> str:
    """Resolve a path string: relative -> absolute via ComfyUI base.

    Bare command names (no path separators) are returned as-is so
    the OS can resolve them via PATH - this is how llama-server
    should be found when it's installed globally."""
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


class PromptEnhancer:
    """Enhance a prompt using a local LLM (llama-server).

    Supports multiple target models (LTX 2.3, KREA-2) via presets.
    The target model is determined by the selected preset.

    This node:
    1. Unloads all ComfyUI models to free VRAM
    2. Spawns llama-server with your GGUF model
    3. Sends your prompt + system preset to the LLM
    4. Retries (with different seeds) until a quality prompt is returned
    5. Kills the server when done

    Each execution produces a unique, enhanced prompt.
    Include any LoRA info or extra context directly in the prompt field.
    """

    # Class-level cache for preset options (built once at first INPUT_TYPES call)
    _preset_display_names: list[str] | None = None
    _display_to_key: dict[str, str] | None = None

    @classmethod
    def _ensure_preset_options(cls):
        if cls._preset_display_names is None:
            cls._preset_display_names, cls._display_to_key = _build_preset_options()

    @classmethod
    def INPUT_TYPES(s):
        s._ensure_preset_options()
        default_display = _get_default_preset_display(s._display_to_key)

        return {
            "required": {
                "preset": (
                    s._preset_display_names if s._preset_display_names else ["no presets found"],
                    {
                        "tooltip": (
                            "System prompt preset that guides how the LLM enhances your prompt.\n"
                            "The target model (LTX T2V/I2V, KREA 2) is determined by the preset."
                        ),
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": (
                            "Describe the scene you want to generate...\n\n"
                            "Example: A woman walking through a rainy street at night\n\n"
                            "You can also include extra context like:\n"
                            "Here is additional info from the used loras: character has pink hair, blue eyes"
                        ),
                    },
                ),
                "llm_model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_PATH,
                        "placeholder": (
                            "Full path to your .gguf model file\n"
                            "Example: C:/models/qwen2.5-7b-instruct-q4_k_m.gguf"
                        ),
                        "tooltip": "Path to the GGUF model file for llama-server",
                    },
                ),
                "llama_server_path": (
                    "STRING",
                    {
                        "default": DEFAULT_SERVER_PATH,
                        "placeholder": "llama-server or full path to llama-server binary",
                        "tooltip": "Path to the llama-server executable (or 'llama-server' if in PATH)",
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Higher = more creative/varied. Lower = more focused",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Nucleus sampling cutoff. Lower = more focused vocabulary",
                    },
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 64,
                        "min": 1,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Keep only top K tokens at each step. Lower = more deterministic",
                    },
                ),
                "min_p": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Minimum probability threshold. Filters out very unlikely tokens",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**32 - 1,
                        "step": 1,
                        "tooltip": (
                            "Random seed base. -1 = random each run (unique prompts). "
                            "Fixed seed = reproducible. Retries increment this seed."
                        ),
                    },
                ),
                "max_retries": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 20,
                        "step": 1,
                        "tooltip": (
                            "Max attempts to generate a quality prompt. "
                            "Each retry uses a different seed. Server stays alive across retries."
                        ),
                    },
                ),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional reference image sent to the LLM alongside your prompt. "
                            "Useful for I2V presets: the LLM sees the image and tailors the prompt to match."
                        ),
                    },
                ),
                "extra_flags": (
                    "STRING",
                    {
                        "default": DEFAULT_EXTRA_FLAGS,
                        "placeholder": "Extra llama-server flags: --no-mmap --threads 12 -ngl 99",
                        "tooltip": "Additional flags passed to llama-server",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "execute"
    CATEGORY = "Prompt Enhancer"
    DESCRIPTION = (
        "Enhance prompts using a local LLM via llama-server. "
        "Supports LTX 2.3 (T2V/I2V) and KREA-2 (T2I) via presets. "
        "Unloads models, runs LLM with retry loop, returns unique enhanced prompt."
    )

    def execute(
        self,
        prompt: str,
        preset: str,
        llm_model_path: str,
        llama_server_path: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        min_p: float = 0.01,
        seed: int = -1,
        max_retries: int = 5,
        extra_flags: str = DEFAULT_EXTRA_FLAGS,
        reference_image = None,
    ):
        if not prompt.strip():
            return ("",)

        # Resolve preset: handle both display names and legacy keys
        preset_key = preset
        if self._display_to_key and preset in self._display_to_key:
            preset_key = self._display_to_key[preset]

        # Load preset system prompt
        system_prompt = load_preset(preset_key)
        if not system_prompt:
            # Fallback: try the value as-is (legacy workflow compat)
            system_prompt = load_preset(preset)
        if not system_prompt:
            # Last resort: try default
            default = get_default_preset()
            if default:
                system_prompt = load_preset(default.key)
        if not system_prompt:
            return (prompt,)

        # User may include LoRA info or any extra context directly in their prompt text.
        # The LLM handles it naturally - no special injection needed.
        user_prompt = prompt.strip()

        # Resolve paths
        model_path = _resolve_path(llm_model_path)
        server_path = _resolve_path(llama_server_path, DEFAULT_SERVER_PATH)

        # Use None for seed if -1 (random)
        seed_value = None if seed == -1 else seed

        # Call the LLM client (handles retry loop internally)
        result = enhance_prompt(
            server_path=server_path,
            model_path=model_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extra_flags=extra_flags,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed_value,
            max_retries=max_retries,
            image=reference_image,
        )

        # Return enhanced prompt, or original on failure
        return (result if result else prompt,)


class PromptEnhancerBatch:
    """Enhance multiple prompts using a local LLM.

    Each prompt gets its own server session with quality validation and retry loop.
    """

    _preset_display_names: list[str] | None = None
    _display_to_key: dict[str, str] | None = None

    @classmethod
    def _ensure_preset_options(cls):
        if cls._preset_display_names is None:
            cls._preset_display_names, cls._display_to_key = _build_preset_options()

    @classmethod
    def INPUT_TYPES(s):
        s._ensure_preset_options()
        default_display = _get_default_preset_display(s._display_to_key)

        return {
            "required": {
                "preset": (
                    s._preset_display_names if s._preset_display_names else ["no presets found"],
                    {
                        "tooltip": (
                            "System prompt preset. Target model determined by preset."
                        ),
                    },
                ),
                "prompts": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": (
                            "Enter one prompt per line:\n\n"
                            "Prompt 1\nPrompt 2\nPrompt 3"
                        ),
                    },
                ),
                "llm_model_path": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_PATH,
                        "placeholder": "Full path to your .gguf model file",
                    },
                ),
                "llama_server_path": (
                    "STRING",
                    {"default": DEFAULT_SERVER_PATH},
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 64, "min": 1, "max": 200, "step": 1},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2**32 - 1, "step": 1},
                ),
                "max_retries": (
                    "INT",
                    {"default": 3, "min": 1, "max": 20, "step": 1},
                ),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional reference image sent to the LLM alongside each prompt."
                        ),
                    },
                ),
                "extra_flags": (
                    "STRING",
                    {"default": DEFAULT_EXTRA_FLAGS},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompts",)
    FUNCTION = "execute"
    CATEGORY = "Prompt Enhancer"
    DESCRIPTION = (
        "Enhance multiple prompts (one per line) with retry loop. "
        "Each prompt gets its own server session with quality validation."
    )

    def execute(
        self,
        prompts: str,
        preset: str,
        llm_model_path: str,
        llama_server_path: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        min_p: float = 0.01,
        seed: int = -1,
        max_retries: int = 3,
        extra_flags: str = DEFAULT_EXTRA_FLAGS,
        reference_image = None,
    ):
        # Parse prompts (one per line)
        raw_prompts = [p.strip() for p in prompts.split("\n") if p.strip()]
        if not raw_prompts:
            return ("",)

        # Resolve preset
        preset_key = preset
        if self._display_to_key and preset in self._display_to_key:
            preset_key = self._display_to_key[preset]

        # Load preset
        system_prompt = load_preset(preset_key)
        if not system_prompt:
            system_prompt = load_preset(preset)
        if not system_prompt:
            default = get_default_preset()
            if default:
                system_prompt = load_preset(default.key)
        if not system_prompt:
            return ("\n".join(raw_prompts),)

        # Resolve paths
        model_path = _resolve_path(llm_model_path)
        server_path = _resolve_path(llama_server_path, DEFAULT_SERVER_PATH)
        seed_value = None if seed == -1 else seed

        # Enhance each prompt
        results = []
        for i, prompt in enumerate(raw_prompts):
            print(f"  [PromptEnhancer Batch] Enhancing prompt {i+1}/{len(raw_prompts)}...")

            # Each prompt gets a unique seed offset
            prompt_seed = None if seed_value is None else seed_value + (i * 1000)

            result = enhance_prompt(
                server_path=server_path,
                model_path=model_path,
                system_prompt=system_prompt,
                user_prompt=prompt,
                extra_flags=extra_flags,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                seed=prompt_seed,
                max_retries=max_retries,
                image=reference_image,
            )
            results.append(result if result else prompt)

        return ("\n".join(results),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PromptEnhancer": PromptEnhancer,
    "PromptEnhancerBatch": PromptEnhancerBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptEnhancer": "Prompt Enhancer",
    "PromptEnhancerBatch": "Prompt Enhancer (Batch)",
}
