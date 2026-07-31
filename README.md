# ComfyUI Prompt Enhancer

LLM-powered prompt enhancement using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this into your `custom_nodes/` directory. No API keys needed - runs entirely local.

## Supported Models

- **KREA 2 T2I** - text-to-image prompt expansion based on [Krea-2's official `expansion.txt`](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt) and community research
- **LTX 2.3 10Eros I2V** - image-to-video motion prompt engineering based on [LTX 2.3 official prompt guide](https://ltx.io/blog/ltx-2-3-prompt-guide) and community research

## Features

- **Auto-detecting presets**: 2 built-in presets that handle both SFW and NSFW content automatically via in-prompt directives
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
| `prompt` | String | Your base prompt. The preset auto-detects SFW vs NSFW content |
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
| `KREA 2 T2I` | KREA 2 Text-to-Image | Prompt expansion for both SFW and NSFW content. Based on [official Krea-2 guidelines](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt) and [community research](https://civitai.com/models/2749367) |
| `LTX 2.3 10Eros I2V` | LTX 2.3 Image-to-Video | Motion prompt engineering for both SFW and NSFW content. Based on [official LTX 2.3 guide](https://ltx.io/blog/ltx-2-3-prompt-guide) and [community research](https://huggingface.co/TenStrip/LTX2.3-10Eros_Workflows) |

Each preset contains both general and NSFW-specific directives. The LLM detects the content type from your prompt and applies the appropriate rules automatically - no need to switch presets.

### Supported LoRAs (LTX 2.3 10Eros I2V)

The LTX preset includes an intelligent LoRA routing guide. The LLM analyzes your prompt and activates the relevant LoRA sections based on keyword matching. Multiple LoRAs can activate from a single prompt.

| LoRA File | Activates on | Purpose |
|-----------|-------------|---------|
| `LTX2.3_reasoning_Sulphur-2_I2V_V4` | **Always** | Universal prompt-following and motion precision. Stacks with everything. |
| `LTX2.3_Physics_V2` | thrust, slam, impact, bounce, collision, momentum, grind, pound | Grounded body-to-body contact with real impact and friction. |
| `Cr3ampi3_animation_sulphur-2_i2v_v1.0` | creampie, cum inside, fills her, breeding, loaded, internal ejaculation | Internal ejaculation animation with fluid temporal behavior and body reactions. |
| `throat_bulge-10Eros_i2v_v1.0` | deepthroat, throat bulge, swallows cock, takes it deep, full swallow | Visible throat deformation during deepthroat with muffled audio and head movement. |
| `ltx23-ultimatedt-NSFW-sulphured_audio_final_k3nk` | blowjob, oral, sucking, cock, penis, dick, testicles | General NSFW refinement with better penis anatomy and sulphur audio integration. |
| `LTX2_3_NSFW_furry_concat_v2` | anthro, furry, anthropomorphic, snout, fur, tail, paws | Multi-purpose NSFW for furry and non-furry content. Supports 2D, 3D, and realistic styles. |
| `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe` | N/A (technical) | Faster generation with fewer steps. Loaded automatically by workflow. |

LoRAs not listed above (e.g. `gemma-3-12b-it-abliterated`, `ltx-2.3-22b-distilled-lora-384-1.1`) are available in the models folder but not recommended - see the preset for details.

### Custom Presets

Add `.txt` files to the `presets/` directory. Use the naming convention `<target>-<name>.txt` (e.g. `krea2-t2i-portrait.txt`). The target prefix determines the display label in the dropdown.

## Reference Images (Multimodal)

For I2V workflows, connect a reference image so the LLM can see the source frame and tailor the motion prompt to match. Requires:

- A multimodal GGUF model (e.g. `gemma-4-31B-it-QAT`)
- The multimodal projector in `extra_flags`:
  ```
  --mmproj "C:/path/to/mmproj-gemma-4-31B-it-QAT-BF16.gguf"
  ```

## Extra Flags Reference

The `extra_flags` input passes arguments directly to `llama-server`. Here are the flags used in the example workflows:

| Flag | Description |
|------|-------------|
| `--no-mmap` | Disable memory-mapping the model file. Only use when the model fits in VRAM easily - avoids disk I/O during generation. |
| `--threads N` | CPU threads for generation. Set to ~75% of physical (performance) cores - not hyperthreaded/logical threads. |
| `-c N` / `--ctx-size N` | Prompt context window size in tokens. `16000` gives the LLM enough headroom for the system prompt, user prompt, and generation without truncation. |
| `--mmproj PATH` | Path to the multimodal projector GGUF file. Required for LTX I2V (reference image input). Optional for KREA 2 T2I - use as a visual hint for the LLM. Must match the base model (e.g. `mmproj-gemma-4-31B-it-*.gguf` for Gemma 4 31B). |
| `--model-draft PATH` + `--spec-type draft-mtp` | Speculative decoding for generation speedup. A draft model pre-generates candidate tokens that the main model accepts or rejects in parallel. Only adds value if your hardware has headroom to run both models. `draft-mtp` uses Multi-Token Prediction (requires an MTP-trained draft model like Unsloth's `gemma-4-31B-it-MTP-BF16.gguf`). |

### KREA 2 T2I (text-only)

```text
--no-mmap --threads 12 -c 16000 \
  --model-draft "C:\Users\maxkr\.lmstudio\models\unsloth\gemma-4-31B-it-GGUF\gemma-4-31B-it-MTP-BF16.gguf" \
  --spec-type draft-mtp
```

No `--mmproj` by default - optional if you want to send a reference image as a visual hint to the LLM. Speculative decoding with the MTP draft model speeds up generation if your hardware has capacity.

### LTX 2.3 10Eros I2V (multimodal)

```text
--no-mmap --threads 12 -c 16000 \
  --mmproj "C:\Users\maxkr\.lmstudio\models\lmstudio-community\gemma-4-31B-it-QAT-GGUF\mmproj-gemma-4-31B-it-NVFP4-turbo-bf16.gguf" \
  --model-draft "C:\Users\maxkr\.lmstudio\models\unsloth\gemma-4-31B-it-GGUF\gemma-4-31B-it-MTP-BF16.gguf" \
  --spec-type draft-mtp
```

Adds `--mmproj` for vision input so the LLM can see the reference image. Same speculative decoding setup.

## Example Workflows

### KREA 2 Text-to-Image

```
[CLIP Text Encode] -> [Prompt Enhancer (preset: KREA 2 T2I)] -> [KREA 2 Sampler]
```

The preset handles both SFW and NSFW content automatically based on your prompt.

### LTX 2.3 Image-to-Video

```
[Load Image] -> [Prompt Enhancer (preset: LTX 2.3 10Eros I2V, reference_image connected)]
                        -> [PreviewAny] -> [CLIP Text Encode] -> [LTX 2.3 Sampler]
```

The reference image lets the LLM see the source frame and describe motion relative to what's already visible. The preset handles both SFW and NSFW content automatically.

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
- **Refusal outputs**: Try a less-aligned model or increase temperature
- **Slow generation**: Use a smaller model or add `-ngl 99` for full GPU offload
- **Reference image not working**: Make sure your model is multimodal and you passed `--mmproj` in extra_flags
