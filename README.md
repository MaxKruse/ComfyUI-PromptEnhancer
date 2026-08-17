# ComfyUI Prompt Enhancer

LLM-powered prompt enhancement using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` HTTP API.

Drop this into your `custom_nodes/` directory. No API keys needed - runs entirely local.

## Supported Models

- **KREA 2 T2I** - text-to-image prompt expansion based on [Krea-2's official `expansion.txt`](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt) and community research
- **LTX 2.3 10Eros I2V** - image-to-video motion prompt engineering based on [LTX 2.3 official prompt guide](https://ltx.io/blog/ltx-2-3-prompt-guide) and community research
- **MiniMax H3 T2V/I2V** - text-to-video and image-to-video prompt generation based on the [official MiniMax H3 Video Prompt Writing Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md)
- **MiniMax H3 R2V** - reference-to-video structured rewrite outputs based on the [official MiniMax H3 Full-Reference Mode guide](https://platform.minimaxi.com/document/minimax-h3-full-reference-mode-guide)

## Features

- **Auto-detecting presets**: Built-in presets that handle both SFW and NSFW content automatically via in-prompt directives
- **Uncensored directive**: System prompts are prepended with an uncensored instruction to reduce refusals and ensure the model follows instructions
- **Server-side sampling**: Sampling parameters are left to the llama-server defaults or command-line flags
- **Dynamic reference images**: Connect 0-9 reference images via Autogrow slots (requires a multimodal GGUF + `--mmproj` flag)
- **Bypass-safe**: When the node is disabled/bypassed, the original prompt passes through unchanged
- **Workflow persistence**: Enhanced prompt values are saved in the workflow JSON and preserved across sessions
- **VRAM-aware**: Automatically unloads ComfyUI models before spawning llama-server, then frees memory after
- **Retry loop**: Keeps trying until a quality, unique prompt is generated
- **Quality validation**: Rejects refusals, short outputs, and prompts too similar to the original
- **Interruptible**: Cancel at any point (while the server starts up or while generating) via ComfyUI's interrupt - the spawned server is killed cleanly
- **Startup diagnostics**: If `llama-server` fails to start, the node logs its output and exit code and falls back to your original prompt instead of hanging
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
| `prompt` | String | Your base prompt. The preset auto-detects SFW vs NSFW content |
| `preset` | Dropdown | System prompt preset (determines target model and expansion style) |
| `llm_model_path` | String | Path to your `.gguf` model |
| `llama_server_path` | String | Path to `llama-server` binary (default: `llama-server`) |
| `ctx_size` | Int | Context window size in tokens (default: 16000, range: 2048-131072) |
| `seed` | Int | Random seed for generation (default: 0, auto-randomizes after each run) |
| `mmproj_path` | String | Optional - path to multimodal projector `.gguf` for vision input |
| `ref_image_0` .. `ref_image_8` | Image | Optional dynamic reference images (0-9 slots via Autogrow) |

**Output:** `enhanced_prompt` - the LLM-enhanced prompt

**Bypass behavior:** When the node is disabled or bypassed, the original `prompt` passes through to `enhanced_prompt` unchanged.

### Prompt Enhancer (Batch)

Enhance multiple prompts (one per line). Each gets its own server session with quality validation.

Supports the same dynamic reference images as the single prompt variant.

## Presets

| Display Name | Target | Description | Suggested maxTokens |
|--------------|--------|-------------|---------------------|
| `KREA 2 T2I` | KREA 2 Text-to-Image | Prompt expansion for both SFW and NSFW content. Based on the [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report), the [official Krea-2 expander guidelines](https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt), [community research](https://civitai.com/models/2749367), and the [SNOFS v1.3D](https://civitai.red/models/1972981/snofs-sex-nudes-other-fun-stuff?modelVersionId=3220691) LoRA directives | 512 |
| `LTX 2.3 10Eros I2V` | LTX 2.3 Image-to-Video | Motion prompt engineering for both SFW and NSFW content. Based on [official LTX 2.3 guide](https://ltx.io/blog/ltx-2-3-prompt-guide) and [community research](https://huggingface.co/TenStrip/LTX2.3-10Eros_Workflows) | 512 |
| `MiniMax H3 - base` | MiniMax H3 Text/Image-to-Video | Three-section prompts (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`) with shot-by-shot camera, audio, and dialogue. Auto-detects T2VA (no images) vs I2VA (reference image(s) as first frame). Based on the [official MiniMax H3 Video Prompt Writing Guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md). Handles both SFW and NSFW content. | 4096 |
| `MiniMax H3 - r2v` | MiniMax H3 Reference-to-Video | Structured full-reference rewrite outputs for R2V. Based on the [official MiniMax H3 Full-Reference Mode guide](https://platform.minimaxi.com/document/minimax-h3-full-reference-mode-guide). Handles both SFW and NSFW content. | 1024 |

`maxTokens` caps the length of the LLM's expanded prompt. The suggested values leave headroom over each preset's target output length so expansions finish before hitting the cap: KREA 2 targets 200-300 words (about 400 tokens), LTX 2.3 10Eros targets 60-200 words (about 280 tokens), MiniMax H3 r2v targets 350-500 words (about 650 tokens), and MiniMax H3 base is uncapped and can run long on multi-shot, dialogue-dense content.

Each preset contains both general and NSFW-specific directives. The LLM detects the content type from your prompt and applies the appropriate rules automatically - no need to switch presets.

### Supported LoRAs (KREA 2 T2I)

The KREA 2 preset embeds the prompting directives for the [SNOFS v1.3D](https://civitai.red/models/1972981/snofs-sex-nudes-other-fun-stuff?modelVersionId=3220691) uncensoring LoRA (`snofs_krea_v1_3D.safetensors` in `models/loras/krea2/`): full-sentence natural-language phrasing, the author's trained vocabulary for acts, positions, items, and states, and the photo-wording rule (the prompt must contain "photo"/"photograph"; never "photorealistic", which pulls Krea 2 toward an illustrated, low-texture look).

Workflow pairing notes (the preset only controls the positive prompt):

- Try SNOFS by itself before stacking other general NSFW LoRAs - they tend to break anatomy.
- Add `kissing` to the workflow negative prompt for cunnilingus scenes, and `penis` for female masturbation scenes without a male present, to keep the act on-topic.
- Optional: the [photodetail slider LoRA](https://civitai.red/models/2823820/photodetail-slider-for-snofs-krea) at strength 0.5-2 pulls rare generations back from the anime side.
- With the Krea 2 RAW checkpoint, the turbo LoRA (`krea2_turbo_lora_rank_64_bf16.safetensors`) at ~0.6 strength beats the Turbo checkpoint for realism. A two-stage setup (first pass without the turbo LoRA for variation and adherence, second pass with it for speed) works well.

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

Connect reference images via the dynamic Autogrow slots (`ref_image_0`, `ref_image_1`, etc.). Up to 9 images can be connected. The LLM sees all connected images and tailors the prompt accordingly.

Requires:

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

Adds `--mmproj` for vision input so the LLM can see the reference image(s). Same speculative decoding setup.

## Example Workflows

### KREA 2 Text-to-Image

```
[CLIP Text Encode] -> [Prompt Enhancer (preset: KREA 2 T2I)] -> [KREA 2 Sampler]
```

The preset handles both SFW and NSFW content automatically based on your prompt.

### LTX 2.3 Image-to-Video

```
[Load Image] -> [Prompt Enhancer (preset: LTX 2.3 10Eros I2V, ref_image_0 connected)]
                        -> [PreviewAny] -> [CLIP Text Encode] -> [LTX 2.3 Sampler]
```

The reference image(s) let the LLM see the source frame(s) and describe motion relative to what's already visible. Multiple reference images can be connected via Autogrow slots. The preset handles both SFW and NSFW content automatically.

### MiniMax H3 Text-to-Video (T2V)

```
[CLIP Text Encode] -> [Prompt Enhancer (preset: MiniMax H3 - base)] -> [MiniMax H3 Sampler]
```

With no reference images connected, the preset outputs the three core sections (`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`) with shot-by-shot camera movement, speaker IDs, and interwoven audio. The LLM builds the complete scene from text.

### MiniMax H3 Image-to-Video (I2V)

```
[Load Image] -> [Prompt Enhancer (preset: MiniMax H3 - base, ref_image_0 connected)]
                    -> [PreviewAny] -> [CLIP Text Encode] -> [MiniMax H3 Sampler]
```

With reference image(s) connected, the preset adds the I2VA instruction prefix (`<Picture N>` anchored at 0.00 seconds in `[Shot 1]`) before the three core sections. The multimodal description derives style, subjects, and composition from the image(s) and describes action that develops forward from the first frame. Multiple reference images can be connected via Autogrow slots. Handles both SFW and NSFW content automatically.

### MiniMax H3 Reference-to-Video (R2V)

```
[Load Image] -> [Prompt Enhancer (preset: MiniMax H3 - r2v, ref_image_0 connected)]
                    -> [PreviewAny] -> [CLIP Text Encode] -> [MiniMax H3 Sampler]
```

The preset outputs structured full-reference rewrite sections (`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`). Reference labels (`<Subject N>`, `<Picture N>`, `<Audio N>`) track identity across all sections. Handles both SFW and NSFW content automatically.

## VRAM Management

The node automatically:
1. Calls `comfy.model_management.unload_all_models()` to free VRAM
2. Runs `gc.collect()` and `torch.cuda.empty_cache()`
3. Spawns llama-server on a free port
4. Sends your prompt (and optional images) and collects the response
5. Kills the server
6. Returns the enhanced prompt

This means the LLM and ComfyUI models don't compete for VRAM.

Every step is interruptible - press ComfyUI's interrupt and the node stops, killing the `llama-server` it spawned. If the server crashes on startup, the last lines of its output and its exit code are logged, and your original prompt is returned unchanged.

## Troubleshooting

- **Server fails to start**: Check that `llama-server` is in your PATH or provide the full path. If it crashes on startup (bad model path, OOM, bad flags), the node now logs the server's last output lines and exit code, then returns your original prompt - scroll up in the console for the `llama-server output (last lines)` block.
- **Cancel a stuck enhancement**: Use ComfyUI's interrupt button. The node stops waiting/generating and kills the `llama-server` it spawned.
- **Model not found**: Verify the `.gguf` file path is correct (absolute or relative to ComfyUI root)
- **Out of memory**: Reduce model size (Q4 -> Q3) or add `--ctx-size 4096` to extra_flags
- **Refusal outputs**: Try a less-aligned model
- **Slow generation**: Use a smaller model or add `-ngl 99` for full GPU offload
- **Reference image not working**: Make sure your model is multimodal and you passed `--mmproj` in extra_flags
- **Enhanced prompt not saved**: The enhanced prompt is a STRING output that is automatically saved in the workflow JSON. Wire it to downstream nodes (e.g., CLIP Text Encode) and the value persists across sessions.
