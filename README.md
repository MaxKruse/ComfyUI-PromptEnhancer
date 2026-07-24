# ComfyUI Prompt Enhancer

LLM-powered prompt enhancement for **KREA-2** image generation, using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this into your `custom_nodes/` directory. No API keys needed - runs entirely local.

## Features

- **KREA-2 support**: KREA-2 (T2I) target model
- **Single preset dropdown**: Choose your preset - the target model is determined automatically
- **VRAM-aware**: Automatically unloads ComfyUI models before spawning llama-server, then frees memory after
- **Retry loop**: Keeps trying until a quality, unique prompt is generated
- **Quality validation**: Rejects refusals, short outputs, and prompts too similar to the original
- **Sampling control**: `top_p`, `top_k`, `min_p` support for fine-tuned generation
- **Reasoning model support**: `max_tokens` defaults to 4096 to accommodate thinking tokens
- **Preset system**: 2 built-in presets for KREA-2
- **LoRA context**: Include LoRA info directly in your prompt - the LLM incorporates it naturally
- **Batch mode**: Enhance multiple prompts sequentially

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/.../ComfyUI-PromptEnhancer.git
```

Requires:
- `llama-server` binary in your PATH (or specify full path in the node)
- A GGUF model file (e.g., Qwen2.5, Llama 3, Mistral)

## Nodes

### Prompt Enhancer

Single prompt enhancement with retry loop.

**Inputs:**
| Input | Type | Description |
|-------|------|-------------|
| `preset` | Dropdown | System prompt preset (includes target model in display name) |
| `prompt` | String | Your base prompt. Include any extra context (LoRA info, style notes, etc.) |
| `llm_model_path` | String | Path to your `.gguf` model |
| `llama_server_path` | String | Path to `llama-server` binary (default: `llama-server`) |
| `temperature` | Float | 0.1-2.0 (default: 1.0) |
| `top_p` | Float | 0.1-1.0 (default: 0.95) |
| `top_k` | Int | 1-200 (default: 64) |
| `min_p` | Float | 0.0-1.0 (default: 0.01) |
| `max_retries` | Int | 1-20 (default: 5) |
| `reference_image` | Image | Optional reference image |
| `extra_flags` | String | Extra llama-server flags (e.g., `-ngl 99 --mlock`) |

**Output:** `enhanced_prompt` - the LLM-enhanced prompt

### Prompt Enhancer (Batch)

Enhance multiple prompts (one per line). Each gets its own server session with quality validation.

## Presets

The preset dropdown shows all available presets with clear labels that include the target model:

| Display Name | Target Model | Description |
|--------------|--------------|-------------|
| `KREA 2 T2I - General` | KREA-2 Text-to-Image | Detailed image prompts based on [KREA-2 guidelines](https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md) |
| `KREA 2 T2I - NSFW` | KREA-2 Text-to-Image | Explicit, uncensored text-to-image prompts |

Custom presets: Add `.txt` files to the `presets/` directory. Use the naming convention `<target>-<name>.txt` (e.g., `krea2-t2i-portrait.txt`).

## LoRA Context

Include LoRA trigger words, style info, or character details directly in your prompt:

```
A woman walking through a rainy street at night

Here is additional info from the used loras:
Character has pink twin-tails, blue eyes, wears a red school uniform
```

The LLM naturally incorporates this context into the enhanced prompt.

## Example Workflows

### KREA-2 Text-to-Image

```
[CLIP Text Encode] -> [PromptEnhancer (preset: KREA 2 T2I - General)] -> [KREA-2 Sampler]
```

For NSFW content, use the `KREA 2 T2I - NSFW` preset with an uncensored checkpoint and NSFW LoRA (e.g., `krea-2-nsfw-v2`).

## VRAM Management

The node automatically:
1. Calls `comfy.model_management.unload_all_models()` to free VRAM
2. Runs `gc.collect()` and `torch.cuda.empty_cache()`
3. Spawns llama-server on a free port
4. Sends your prompt and collects the response
5. Kills the server
6. Returns the enhanced prompt

This means the LLM and ComfyUI models don't compete for VRAM.

## Troubleshooting

- **Server fails to start**: Check that `llama-server` is in your PATH or provide the full path
- **Model not found**: Verify the `.gguf` file path is correct (absolute or relative to ComfyUI root)
- **Out of memory**: Reduce model size (Q4_K_M -> Q3_K_M) or add `--ctx-size 4096` to extra_flags
- **Refusal outputs**: Try a different preset or lower temperature
- **Slow generation**: Use a smaller model or add `-ngl 99` for full GPU offload
