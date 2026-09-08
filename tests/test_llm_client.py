"""Tests for llm_client interruptibility and early server-startup failure handling."""

import logging
import sys
import time
import types

import pytest


class FakeProc:
    """Minimal stand-in for subprocess.Popen exposing poll()."""

    def __init__(self, poll_result):
        self._poll_result = poll_result

    def poll(self):
        return self._poll_result


def _install_fake_comfy(monkeypatch, raise_interrupt=None):
    """Install a fake comfy.model_management into sys.modules.

    If *raise_interrupt* is given (exception class), throw_exception_if_processing_interrupted
    raises it (simulating a pending interrupt). Otherwise it is a no-op.
    """
    fake_comfy = types.ModuleType("comfy")
    fake_mm = types.ModuleType("comfy.model_management")

    if raise_interrupt is None:
        fake_mm.throw_exception_if_processing_interrupted = lambda: None
    else:
        exc = raise_interrupt

        def _throw():
            raise exc() if isinstance(exc, type) else exc

        fake_mm.throw_exception_if_processing_interrupted = _throw

    fake_comfy.model_management = fake_mm
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_mm)


def test_wait_for_server_detects_early_exit(monkeypatch):
    """wait_for_server should return False quickly once the server process has exited."""
    import llm_client

    _install_fake_comfy(monkeypatch)  # no interrupt pending

    start = time.monotonic()
    result = llm_client.wait_for_server("http://127.0.0.1:1", FakeProc(poll_result=7), timeout=30)
    elapsed = time.monotonic() - start

    assert result is False
    assert elapsed < 1.5, f"should detect early exit without waiting the timeout, took {elapsed:.2f}s"


def test_wait_for_server_raises_on_interrupt(monkeypatch):
    """wait_for_server should raise when ComfyUI requests an interrupt while waiting."""
    import llm_client

    class Interrupt(BaseException):
        pass

    _install_fake_comfy(monkeypatch, raise_interrupt=Interrupt)

    start = time.monotonic()
    with pytest.raises(Interrupt):
        llm_client.wait_for_server("http://127.0.0.1:1", FakeProc(poll_result=None), timeout=30)
    elapsed = time.monotonic() - start
    assert elapsed < 1.5, f"should interrupt quickly, took {elapsed:.2f}s"


def test_chat_completion_interruptible_raises_on_interrupt(monkeypatch):
    """A single in-flight generation should be interruptible mid-request."""
    import llm_client

    class Interrupt(BaseException):
        pass

    _install_fake_comfy(monkeypatch, raise_interrupt=Interrupt)
    # Simulate a slow generation that would otherwise block for 5s.
    monkeypatch.setattr(llm_client, "chat_completion", lambda **kw: (time.sleep(5), None)[1])

    start = time.monotonic()
    with pytest.raises(Interrupt):
        llm_client._run_chat_completion_interruptible("http://127.0.0.1:1", "m", "sys", "user", None, 0)
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"should interrupt before the 5s generation finished, took {elapsed:.2f}s"


def test_start_llama_server_interrupt_kills_server(monkeypatch):
    """An interrupt while the server is starting up must kill the spawned process."""
    import llm_client

    class Interrupt(BaseException):
        pass

    _install_fake_comfy(monkeypatch, raise_interrupt=Interrupt)  # interrupts immediately

    created = []
    real_popen = llm_client.subprocess.Popen

    def recording_popen(*a, **k):
        p = real_popen(*a, **k)
        created.append(p)
        return p

    monkeypatch.setattr(llm_client.subprocess, "Popen", recording_popen)
    # A "server" that never becomes healthy (simulates a slow/ongoing model load).
    monkeypatch.setattr(llm_client, "build_command", lambda *a, **k: [sys.executable, "-c", "import time; time.sleep(60)"])

    with pytest.raises(Interrupt):
        llm_client._start_llama_server("fake", "model", 1, 16000, "", "")

    assert len(created) == 1
    proc = created[0]
    proc.wait(timeout=5)
    assert proc.poll() is not None, "spawned llama-server should be killed on interrupt"


def test_print_safe_survives_non_utf8_console(monkeypatch):
    """_print_safe must not crash when the console codepage can't encode the status symbols."""
    import llm_client

    written = []

    class StrictStream:
        encoding = "cp1252"

        def write(self, s):
            for ch in ("\u2713", "\u2717"):
                if ch in s:
                    raise UnicodeEncodeError("cp1252", s, 0, 1, "charmap")
            written.append(s)
            return len(s)

    monkeypatch.setattr(sys, "stdout", StrictStream())
    llm_client._print_safe("  [PromptEnhancer] \u2713 Accepted on attempt 1 (50 chars)")
    assert "".join(written).strip() != "", "_print_safe should still emit output without raising"


def test_start_llama_server_logs_early_crash(monkeypatch, caplog):
    """When llama-server crashes on startup, fail fast and capture its output."""
    import llm_client

    # Bound the fallback path so the test is fast even if early-exit detection regresses.
    monkeypatch.setattr(llm_client, "_SERVER_STARTUP_TIMEOUT", 3)
    fake_cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('fake-crash-boom\\n'); sys.exit(7)",
    ]
    monkeypatch.setattr(llm_client, "build_command", lambda *a, **k: list(fake_cmd))

    with caplog.at_level(logging.ERROR, logger="llm_client"):
        start = time.monotonic()
        proc, base_url, _ = llm_client._start_llama_server("server", "model", 1, 16000, "", "")
        elapsed = time.monotonic() - start

    assert proc is None
    assert base_url == ""
    assert elapsed < 2.5, f"early crash should be detected fast (well under the {llm_client._SERVER_STARTUP_TIMEOUT}s timeout), took {elapsed:.2f}s"
    assert "fake-crash-boom" in caplog.text, f"server output not captured:\n{caplog.text}"
    assert "exited early" in caplog.text, f"should report early exit:\n{caplog.text}"


def test_build_command_parses_default_extra_flags():
    """The node's default extra flags (Muse-Glimmer DFlash draft, quoted Windows path)
    must survive shlex.split with the draft path intact."""
    import llm_client
    from nodes import DEFAULT_EXTRA_SERVER_ARGS

    cmd = llm_client.build_command("llama-server", "model.gguf", 1234, extra_flags=DEFAULT_EXTRA_SERVER_ARGS)

    assert cmd[0] == "llama-server"
    assert "--model-draft" in cmd
    draft = cmd[cmd.index("--model-draft") + 1]
    assert draft == "C:\\Users\\maxkr\\LLMs\\Muse-Glimmer\\dflash-kquant.gguf"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dflash"
    assert "--top-p" in cmd and "--top-k" in cmd


def test_chat_completion_sends_seed_in_payload(monkeypatch):
    """chat_completion must send the seed in the request payload so llama-server seeds each request."""
    import json
    import llm_client

    captured = {}

    class FakeResp:
        status = 200

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "a detailed prompt"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr(llm_client.request, "urlopen", fake_urlopen)
    result = llm_client.chat_completion("http://127.0.0.1:1", "model", "system", "user", seed=1234)

    assert result == "a detailed prompt"
    assert captured["payload"]["seed"] == 1234


def test_retry_loop_uses_distinct_seed_per_attempt(monkeypatch):
    """Each retry attempt must use the next seed value, or retries would repeat identical output."""
    import llm_client

    seen = []

    def fake_chat(base_url, model_name, system_prompt, user_prompt, images, seed):
        seen.append(seed)
        return "too short"  # fails min_words -> loop retries

    monkeypatch.setattr(llm_client, "_run_chat_completion_interruptible", fake_chat)
    best = llm_client._run_retry_loop("http://127.0.0.1:1", "model", "system", "user", None, max_retries=3, min_words=50, seed=100)

    assert seen == [100, 101, 102]
    assert best == "too short"


def test_retry_loop_negative_seed_is_random_32bit_base(monkeypatch):
    """seed=-1 must resolve to a random 32-bit base seed that still increments per attempt."""
    import llm_client

    seen = []

    def fake_chat(base_url, model_name, system_prompt, user_prompt, images, seed):
        seen.append(seed)
        return "too short"

    monkeypatch.setattr(llm_client, "_run_chat_completion_interruptible", fake_chat)
    llm_client._run_retry_loop("http://127.0.0.1:1", "model", "system", "user", None, max_retries=2, min_words=50, seed=-1)

    assert all(0 <= s < 2**32 for s in seen)
    assert seen[1] == (seen[0] + 1) % 2**32


def test_build_command_does_not_pass_seed_flag():
    """The seed travels in the chat completion payload, not on the llama-server command line."""
    import llm_client

    cmd = llm_client.build_command("llama-server", "model.gguf", 1234)
    assert "--seed" not in cmd


def test_is_good_prompt_accepts_faithful_expansion():
    """An expansion that preserves every original word but adds real content must pass the gate."""
    import llm_client

    original = "a woman in a red dress walks through a rainy street at night"
    enhanced = (
        "a woman in a red dress walks through a rainy street at night, "
        "neon signs reflected in the wet asphalt, steam rising from a grate, "
        "her umbrella tilted against a hard wind, puddles mirroring the city lights, "
        "cinematic lighting, film grain, shallow depth of field"
    )

    assert llm_client.is_good_prompt(enhanced, original, min_words=20)


def test_is_good_prompt_rejects_echoes():
    """A near-verbatim echo of the original must fail the gate even when long enough."""
    import llm_client

    original = (
        "a woman in a red dress walks through a rainy street at night "
        "neon signs reflected in the wet asphalt steam rising from a grate "
        "her umbrella tilted against a hard wind puddles mirroring the city lights"
    )

    assert not llm_client.is_good_prompt(original + " again", original, min_words=20)


def test_is_good_prompt_rejects_refusal():
    """Refusal or deflection phrasing must fail the gate."""
    import llm_client

    refusal = (
        "I'm sorry, I cannot generate that content for you. Instead I can offer a "
        "general scene of a woman walking through a city street at night with rain, "
        "neon reflections, and ambient light, described in a neutral cinematic style "
        "with no explicit detail, focusing only on the atmosphere and the wet pavement."
    )

    assert not llm_client.is_good_prompt(refusal, "a woman", min_words=20)


def test_retry_loop_never_returns_refusal(monkeypatch):
    """A refusal must never survive as the fallback best result."""
    import llm_client

    def fake_chat(base_url, model_name, system_prompt, user_prompt, images, seed):
        return "I cannot help with that request."

    monkeypatch.setattr(llm_client, "_run_chat_completion_interruptible", fake_chat)
    best = llm_client._run_retry_loop("http://127.0.0.1:1", "model", "system", "user", None, 2, 50, 1)

    assert best is None


def test_retry_loop_prefers_clean_candidate_over_refusal(monkeypatch):
    """When one attempt is a refusal and another is clean but short, the clean one wins."""
    import llm_client

    responses = iter([
        "I cannot assist with that.",
        "a clean but short prompt",
    ])

    def fake_chat(base_url, model_name, system_prompt, user_prompt, images, seed):
        return next(responses)

    monkeypatch.setattr(llm_client, "_run_chat_completion_interruptible", fake_chat)
    best = llm_client._run_retry_loop("http://127.0.0.1:1", "model", "system", "user", None, 2, 50, 1)

    assert best == "a clean but short prompt"


def test_chat_completion_grounds_reference_images_with_labels(monkeypatch):
    """Each reference image must be preceded by a text label so <Picture N> indices are grounded."""
    import json
    import llm_client

    captured = {}

    class FakeResp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr(llm_client.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(llm_client, "_images_to_base64_jpegs", lambda images: ["AAA", "BBB"])

    llm_client.chat_completion("http://127.0.0.1:1", "model", "system", "user", seed=1, images=[object(), object()])

    content = captured["payload"]["messages"][1]["content"]
    assert [part["type"] for part in content] == ["text", "text", "image_url", "text", "image_url"]
    assert content[0]["text"] == "user"
    assert "<Picture 1>" in content[1]["text"]
    assert content[2]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
    assert "<Picture 2>" in content[3]["text"]
    assert content[4]["image_url"]["url"] == "data:image/jpeg;base64,BBB"


def test_chat_completion_prints_inference_speed(monkeypatch, capsys):
    """Per-attempt llama-server speed (tok/s) and context usage must be printed to the console."""
    import json
    import llm_client

    payload = {
        "choices": [{"message": {"content": "a detailed prompt"}}],
        "timings": {
            "cache_n": 236,
            "prompt_n": 100,
            "prompt_ms": 500.0,
            "prompt_per_token_ms": 4.17,
            "prompt_per_second": 200.0,
            "predicted_n": 350,
            "predicted_ms": 8750.0,
            "predicted_per_token_ms": 25.0,
            "predicted_per_second": 40.0,
        },
        "usage": {"completion_tokens": 350, "prompt_tokens": 336, "total_tokens": 686},
    }

    class FakeResp:
        def read(self):
            return json.dumps(payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        return FakeResp()

    monkeypatch.setattr(llm_client.request, "urlopen", fake_urlopen)
    result = llm_client.chat_completion("http://127.0.0.1:1", "model", "system", "user", seed=1)

    assert result == "a detailed prompt"
    out = capsys.readouterr().out
    assert "40.0 tok/s" in out
    assert "350" in out
    assert "ctx used 686" in out


def test_chat_completion_reports_context_overflow(monkeypatch, capsys):
    """A context-overflow HTTP error must be printed to the console with an actionable hint."""
    import io
    import llm_client

    def fake_urlopen(req, timeout=None):
        body = b'{"error": {"message": "Prompt tokens (16500) exceeds remaining slot context (16384)"}}'
        raise llm_client.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(llm_client.request, "urlopen", fake_urlopen)
    result = llm_client.chat_completion("http://127.0.0.1:1", "model", "system", "user", seed=1)

    assert result is None
    out = capsys.readouterr().out
    assert "OUT OF CONTEXT" in out


def test_vision_context_note_with_images():
    """When reference images are visible, the system prompt must say so."""
    import llm_client

    note = llm_client._vision_context_note(["jpeg-b64"])
    assert "can see the reference image" in note
    assert "<Picture 1>" in note
    assert "never invent" in note


def test_vision_context_note_without_images():
    """A blind run must be told not to assume frame details."""
    import llm_client

    note = llm_client._vision_context_note(None)
    assert "No reference image is available" in note
    assert "anchor using only what the user" in note
