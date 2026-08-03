"""llama-server HTTP client for LTX prompt enhancement.

Spawns a temporary llama-server instance, sends the prompt via the OpenAI-compatible
/v1/chat/completions endpoint, then kills the server.

Adapted from sd-webui-llama-server-enhance for ComfyUI integration.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import logging
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Lock
from urllib import request, error

logger = logging.getLogger(__name__)

# How long to wait for the server to become healthy (model load time).
_SERVER_STARTUP_TIMEOUT = 120

# Timeout for individual HTTP requests.
_HTTP_TIMEOUT = 5

# Port range to probe for a free port.
_PORT_RANGE = range(49152, 65536)

_print_lock = Lock()


def _print_safe(msg: str):
    """Thread-safe print wrapper."""
    with _print_lock:
        print(msg)


def _tensor_to_base64_jpeg(image_tensor) -> str:
    """Convert a ComfyUI IMAGE tensor to a base64-encoded JPEG string.

    Accepts:
      - Single image: torch.Tensor shape [H, W, 3], values in [0, 1]
      - Batch:        torch.Tensor shape [N, H, W, 3] — uses first image
    """
    import torch

    # Handle batch — take first image
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]

    # Clamp to [0, 1], convert to uint8, move to CPU
    img = image_tensor.clamp(0, 1).mul(255).byte().cpu()

    # Convert to PIL
    from PIL import Image
    pil_img = Image.fromarray(img.numpy(), mode="RGB")

    # Encode as JPEG
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _images_to_base64_jpegs(images) -> list[str]:
    """Convert a list of ComfyUI IMAGE tensors to base64-encoded JPEG strings.

    Each tensor can be a single image [H, W, 3] or batch [N, H, W, 3].
    Batches use the first image. Returns an empty list if no valid images.
    """
    if not images:
        return []
    return [_tensor_to_base64_jpeg(img) for img in images]


def find_free_port() -> int:
    """Find a free TCP port by probing random ports in the ephemeral range."""
    candidates = random.sample(list(_PORT_RANGE), min(100, len(_PORT_RANGE)))
    for port in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    # Fallback: let the OS pick
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_command(
    server_path: str,
    model_path: str,
    port: int,
    *,
    ctx_size: int = 16000,
    seed: int = 0,
    mmproj_path: str = "",
    extra_flags: str = "",
) -> list[str]:
    """Build the full llama-server command list."""
    import shlex

    user_flags = shlex.split(extra_flags) if extra_flags else []

    cmd = [
        server_path,
        "-m", model_path,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--no-ui",
        "--no-warmup",
        "-c", str(ctx_size),
    ]

    if seed < 0:
        # Use a random seed
        cmd.extend(["--seed", str(random.randint(1, 2**32 - 1))])
    else:
        # llama-server --seed only accepts a 32-bit unsigned int.
        # ComfyUI seeds can be up to 2**63, so clamp to fit.
        cmd.extend(["--seed", str(seed % (2**32))])

    if mmproj_path:
        cmd.extend(["--mmproj", mmproj_path])

    cmd.extend(user_flags)
    return cmd


def wait_for_server(base_url: str, timeout: int = _SERVER_STARTUP_TIMEOUT) -> bool:
    """Poll /health until the server responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    url = f"{base_url}/health"

    while time.monotonic() < deadline:
        try:
            with request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
                if resp.status == 200:
                    return True
        except (error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)

    return False


def discover_model_name(base_url: str) -> str:
    """Fetch the currently loaded model name from /v1/models."""
    try:
        with request.urlopen(f"{base_url}/v1/models", timeout=_HTTP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            for m in models:
                status = m.get("status", {})
                if isinstance(status, dict) and status.get("value") == "loaded":
                    return m.get("id", "llama")
            if models:
                return models[0].get("id", "llama")
    except (error.URLError, json.JSONDecodeError, KeyError):
        pass
    return "llama"


def chat_completion(
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.05,
    images = None,
) -> str | None:
    """Send a single chat completion request. Returns enhanced prompt or None on failure.

    If *images* is a list of ComfyUI IMAGE tensors (torch.Tensor [H,W,3] in [0,1]),
    each is encoded as a base64 JPEG and sent alongside the text prompt so the LLM
    can see all reference images.
    """
    # Build user message content
    if images:
        image_b64s = _images_to_base64_jpegs(images)
        user_content = [
            {"type": "text", "text": user_prompt},
        ]
        for b64 in image_b64s:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    else:
        user_content = user_prompt

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer no-key",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            return content.strip() if content else None
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        _print_safe(f"  [PromptEnhancer] HTTP {exc.code} from server: {body[:500]}")
        logger.error("Chat completion request failed: HTTP %s — %s", exc.code, body[:500])
        return None
    except (error.URLError, json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.error("Chat completion request failed: %s", exc)
        return None


def is_good_prompt(result: str, original: str, min_words: int = 25) -> bool:
    """Validate that an enhanced prompt meets quality thresholds.

    Checks:
    - Has enough words (not a one-liner)
    - Is meaningfully different from the original (not just echoed back)
    - Doesn't contain refusal/hedging language
    """
    if not result:
        return False

    words = result.split()
    if len(words) < min_words:
        return False

    # Check for refusal/hedging patterns
    refusal_patterns = [
        "i cannot", "i can't", "i won't", "i'm not able",
        "i'm sorry", "i apologize", "as an ai", "i am an ai",
        "i cannot assist", "i'm unable", "content policy",
        "sexual content", "explicit content is", "i can help with",
        "instead i can", "here's an alternative",
    ]
    result_lower = result.lower()
    for pattern in refusal_patterns:
        if pattern in result_lower:
            return False

    # Check it's not just the original prompt echoed back
    if result.strip() == original.strip():
        return False

    # Result should be substantially different (at least 30% different tokens)
    orig_words = set(original.lower().split())
    result_words = set(result.lower().split())
    if orig_words and len(orig_words) > 0:
        overlap = len(orig_words & result_words) / len(orig_words)
        if overlap > 0.85:  # More than 85% overlap = probably just echoed
            return False

    return True


def free_gpu_memory():
    """Unload all ComfyUI models and free VRAM for llama-server."""
    import torch
    import comfy.model_management

    _print_safe("  [PromptEnhancer] Freeing GPU memory...")
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache(force=True)
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    gc.collect()
    _print_safe("  [PromptEnhancer] GPU memory freed.")


def kill_server(proc: subprocess.Popen | None):
    """Terminate the server subprocess and wait for it to exit."""
    if proc is None:
        return

    pid = proc.pid
    _print_safe(f"  [PromptEnhancer] Killing llama-server (pid={pid})...")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server (pid=%s) did not terminate, sending SIGKILL", pid)
            proc.kill()
            proc.wait(timeout=3)
    except Exception:
        logger.exception("Error stopping llama-server")
    finally:
        _print_safe(f"  [PromptEnhancer] llama-server (pid={pid}) stopped.")


def enhance_prompt(
    server_path: str,
    model_path: str,
    system_prompt: str,
    user_prompt: str,
    *,
    ctx_size: int = 16000,
    seed: int = 0,
    mmproj_path: str = "",
    extra_flags: str = "",
    temperature: float = 0.9,
    top_p: float = 0.9,
    top_k: int = 40,
    min_p: float = 0.05,
    max_retries: int = 5,
    min_words: int = 25,
    images = None,
) -> str | None:
    """Enhance a prompt by spawning a temporary llama-server instance.

    Retries up to max_retries times until is_good_prompt() passes.
    The server is spawned ONCE and kept alive across retries for efficiency.

    1. Free GPU memory (unload ComfyUI models).
    2. Find a free port and start llama-server.
    3. Wait for /health to confirm ready.
    4. Send the prompt via /v1/chat/completions — retry until good result.
    5. Kill the server.
    6. Return the enhanced prompt text, or None on failure.
    """
    model_path_resolved = Path(model_path).resolve()
    if not model_path_resolved.is_file():
        _print_safe(f"  [PromptEnhancer] ERROR: Model file not found: {model_path}")
        return None

    # Step 1: Free GPU memory
    free_gpu_memory()

    # Step 2: Find port and build command
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = build_command(
        server_path,
        model_path,
        port,
        ctx_size=ctx_size,
        seed=seed,
        mmproj_path=mmproj_path,
        extra_flags=extra_flags,
    )

    _print_safe(f"  [PromptEnhancer] Running llama-server (model: {model_path_resolved.name}, port: {port})...")
    _print_safe(f"  [PromptEnhancer] Command: {' '.join(cmd)}")

    # Windows: prevent console window popup
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    server_proc: subprocess.Popen | None = None
    try:
        server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        _print_safe(f"  [PromptEnhancer] Started llama-server (pid={server_proc.pid})")
    except FileNotFoundError:
        _print_safe(f"  [PromptEnhancer] ERROR: llama-server not found at: {server_path}")
        return None
    except Exception:
        logger.exception("[PromptEnhancer] Failed to start llama-server")
        return None

    # Step 3: Wait for health
    _print_safe(f"  [PromptEnhancer] Waiting for server (timeout: {_SERVER_STARTUP_TIMEOUT}s)...")
    start = time.monotonic()

    if not wait_for_server(base_url, timeout=_SERVER_STARTUP_TIMEOUT):
        elapsed = time.monotonic() - start
        _print_safe(f"  [PromptEnhancer] ERROR: Server did not become healthy within {_SERVER_STARTUP_TIMEOUT}s ({elapsed:.1f}s elapsed)")

        # Log stderr if process crashed
        if server_proc.poll() is not None:
            try:
                stderr = server_proc.stderr.read().decode("utf-8", errors="replace") if server_proc.stderr else ""
                _print_safe(f"  [PromptEnhancer] Server stderr (last 500 chars):\n{stderr[-500:]}")
            except Exception:
                pass

        kill_server(server_proc)
        return None

    ready_time = time.monotonic()
    model_name = discover_model_name(base_url)
    _print_safe(f"  [PromptEnhancer] Server ready in {ready_time - start:.1f}s. Model: {model_name}")

    # Step 4: Retry loop — keep trying until we get a good prompt
    best_result: str | None = None

    for attempt in range(1, max_retries + 1):
        _print_safe(
            f"  [PromptEnhancer] Attempt {attempt}/{max_retries} "
            f"(temp={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p})..."
        )

        result = chat_completion(
            base_url=base_url,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            images=images,
        )

        if result and is_good_prompt(result, user_prompt, min_words=min_words):
            _print_safe(f"  [PromptEnhancer] ✓ Accepted on attempt {attempt} ({len(result)} chars)")
            best_result = result
            break
        elif result:
            _print_safe(f"  [PromptEnhancer] ✗ Rejected attempt {attempt} — quality check failed ({len(result) if result else 0} chars)")
            # Keep as fallback if nothing else works
            if best_result is None:
                best_result = result
        else:
            _print_safe(f"  [PromptEnhancer] ✗ Rejected attempt {attempt} — empty/failed response")

    elapsed = time.monotonic() - ready_time

    # Step 5: Kill server
    # Log stderr if all attempts failed — the server may have crashed on the request
    if best_result is None and server_proc is not None and server_proc.stderr:
        try:
            stderr_raw = server_proc.stderr.read().decode("utf-8", errors="replace")
            if stderr_raw.strip():
                _print_safe(f"  [PromptEnhancer] Server stderr (last 1000 chars):\n{stderr_raw[-1000:]}")
        except Exception:
            pass

    kill_server(server_proc)

    if best_result:
        _print_safe(
            f"  [PromptEnhancer] Final result ({len(best_result)} chars, {elapsed:.1f}s total): "
            f"{best_result[:120]}{'...' if len(best_result) > 120 else ''}"
        )
    else:
        _print_safe(f"  [PromptEnhancer] ERROR: All {max_retries} attempts failed ({elapsed:.1f}s total)")

    return best_result
