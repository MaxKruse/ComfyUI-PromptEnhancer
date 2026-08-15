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
        llm_client._run_chat_completion_interruptible("http://127.0.0.1:1", "m", "sys", "user", None)
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
        llm_client._start_llama_server("fake", "model", 1, 16000, 0, "", "")

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
        proc, base_url, _ = llm_client._start_llama_server("server", "model", 1, 16000, 0, "", "")
        elapsed = time.monotonic() - start

    assert proc is None
    assert base_url == ""
    assert elapsed < 2.5, f"early crash should be detected fast (well under the {llm_client._SERVER_STARTUP_TIMEOUT}s timeout), took {elapsed:.2f}s"
    assert "fake-crash-boom" in caplog.text, f"server output not captured:\n{caplog.text}"
    assert "exited early" in caplog.text, f"should report early exit:\n{caplog.text}"
