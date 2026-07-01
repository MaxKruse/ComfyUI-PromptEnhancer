"""ComfyUI Prompt Enhancer: LLM-powered prompt enhancement for video and image generation.

Uses llama-server (llama.cpp) with GGUF models to enhance prompts via presets.
Supports LTX 2.3 (text-to-video, image-to-video) and KREA-2 (text-to-image) target models.
Unloads ComfyUI models before running the LLM, then frees memory after.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # Allow importing without ComfyUI dependencies (e.g., during testing)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
