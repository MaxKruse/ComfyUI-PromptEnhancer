# ComfyUI Prompt Enhancer

LLM-powered prompt enhancement using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this into your `custom_nodes/` directory. No API keys needed - runs entirely local.

## Supported Models

- **KREA 2 T2I** - text-to-image prompt expansion based on [Krea-2's official `expansion.txt`](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt)
- **LTX 2.3 10Eros I2V** - image-to-video motion prompt engineering with VBVR reasoning LoRA and PhysLTX physics LoRA support

## Features

- **Preset system**: 4 built-in presets (General + NSFW for each target model)
- **Reference image**: Send an image to the LLM alongside your prompt (requires a multimodal GGUF + `--mmproj` flag)
- **VRAM-aware**: Automatically unloads ComfyUI models before spawning llama-server, then frees memory after
- **Retry loop**: Keeps trying until a quality, unique prompt is generated
- **Quality validation**: Rejects refusals, short outputs, and prompts too similar to the original
- **Sampling control**: `temperature`, `top_p`, `top_k`, `min_p` support
- **Batch mode**: Enhance multiple prompts sequentially

## Recommended Models

Large reasoning-capable models give the best results:

- **Gemma 4 31B** (QAT Q4_0, ~17 GB) - excellent prompt following and detail
- **Qwen 3.6 35B** (Q4_K_M, ~20 GB) - strong creative expansion and instruction following
- **Gemma 4 12B** (QAT Q4_0, ~6.6 GB) - good balance of quality and VRAM usage

Any GGUF model works. Smaller models (7B class) are fine for basic prompts but may lack the detail larger models produce.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MaxKruse/ComfyUI-PromptEnhancer.git
```

Requires:
- `llama-server` binary in your PATH (or specify full path in the node)
- A GGUF model file

## Nodes

### Prompt Enhancer

Single prompt enhancement with retry loop.

**Inputs:**

| Input | Type | Description |
|-------|------|-------------|
| `preset` | Dropdown | System prompt preset (determines target model and expansion style) |
| `prompt` | String | Your base prompt. Include any extra context (LoRA info, style notes, etc.) |
| `llm_model_path` | String | Path to your `.gguf` model |
| `llama_server_path` | String | Path to `llama-server` binary (default: `llama-server`) |
| `seed` | Seed | Random seed for generation |
| `temperature` | Float | 0.1-2.0 (default: 1.0) |
| `top_p` | Float | 0.1-1.0 (default: 0.95) |
| `top_k` | Int | 1-200 (default: 64) |
| `min_p` | Float | 0.0-1.0 (default: 0.01) |
| `max_retries` | Int | 1-20 (default: 5) |
| `reference_image` | Image | Optional - sent to multimodal LLMs for visual context |
| `extra_flags` | String | Extra llama-server flags (e.g. `--no-mmap --threads 12 -ngl 99`) |

**Output:** `enhanced_prompt` - the LLM-enhanced prompt

### Prompt Enhancer (Batch)

Enhance multiple prompts (one per line). Each gets its own server session with quality validation.

## Presets

| Display Name | Target | Description |
|--------------|--------|-------------|
| `KREA 2 T2I - General` | KREA 2 Text-to-Image | SFW prompt expansion based on [official Krea-2 guidelines](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt) |
| `KREA 2 T2I - NSFW` | KREA 2 Text-to-Image | Uncensored, explicit text-to-image prompts |
| `LTX 2.3 10Eros I2V - General` | LTX 2.3 Image-to-Video | Cinematic motion prompts with VBVR reasoning and PhysLTX physics support |
| `LTX 2.3 10Eros I2V - NSFW` | LTX 2.3 Image-to-Video | Uncensored, explicit image-to-video motion prompts |

### Custom Presets

Add `.txt` files to the `presets/` directory. Use the naming convention `<target>-<name>.txt` (e.g. `krea2-t2i-portrait.txt`). The target prefix determines the display label in the dropdown.

## Reference Images (Multimodal)

For I2V workflows, connect a reference image so the LLM can see the source frame and tailor the motion prompt to match. Requires:

- A multimodal GGUF model (e.g. `gemma-4-31B-it-QAT`)
- The multimodal projector in `extra_flags`:
  ```
  --mmproj "C:/path/to/mmproj-gemma-4-31B-it-QAT-BF16.gguf"
  ```

## Example Workflows

### KREA 2 Text-to-Image

```
[CLIP Text Encode] -> [Prompt Enhancer (preset: KREA 2 T2I - General)] -> [KREA 2 Sampler]
```

For NSFW content, use the `KREA 2 T2I - NSFW` preset with an uncensored checkpoint and NSFW LoRA.

### LTX 2.3 Image-to-Video

```
[Load Image] -> [Prompt Enhancer (preset: LTX 2.3 10Eros I2V - NSFW, reference_image connected)]
                        -> [PreviewAny] -> [CLIP Text Encode] -> [LTX 2.3 Sampler]
```

The reference image lets the LLM see the source frame and describe motion relative to what's already visible.

## VRAM Management

The node automatically:
1. Calls `comfy.model_management.unload_all_models()` to free VRAM
2. Runs `gc.collect()` and `torch.cuda.empty_cache()`
3. Spawns llama-server on a free port
4. Sends your prompt (and optional image) and collects the response
5. Kills the server
6. Returns the enhanced prompt

This means the LLM and ComfyUI models don't compete for VRAM.

## Troubleshooting

- **Server fails to start**: Check that `llama-server` is in your PATH or provide the full path
- **Model not found**: Verify the `.gguf` file path is correct (absolute or relative to ComfyUI root)
- **Out of memory**: Reduce model size (Q4 -> Q3) or add `--ctx-size 4096` to extra_flags
- **Refusal outputs**: Use the NSFW preset for explicit content, or try a less-aligned model
- **Slow generation**: Use a smaller model or add `-ngl 99` for full GPU offload
- **Reference image not working**: Make sure your model is multimodal and you passed `--mmproj` in extra_flags
