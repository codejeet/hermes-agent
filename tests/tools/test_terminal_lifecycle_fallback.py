"""Exercise lifecycle scanning through terminal_tool's real fallback reader."""

import json
import os
import shlex
import subprocess

import pytest

import tools.terminal_tool as terminal
from cron.lifecycle_guard import _MAX_REFERENCED_SCRIPT_BYTES


class RecordingEnvironment:
    """Only execute read probes against a disposable simulated remote file."""

    env = {}

    def __init__(self, cwd, command, *, remote_path=None, remote_file=None):
        self.cwd = str(cwd)
        self.command = command
        self.remote_path = remote_path
        self.remote_file = remote_file
        self.calls = []
        self.probe_results = []

    def execute(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command == self.command or self.remote_path is None:
            # Never execute the user's binary or script, including blocked scripts.
            return {"output": "command completed", "returncode": 0}

        assert self.remote_path in command
        assert 0 < kwargs.get("timeout", 0) <= 10
        probe = command.replace(
            shlex.quote(self.remote_path), shlex.quote(str(self.remote_file))
        )
        result = subprocess.run(
            probe,
            shell=True,
            capture_output=True,
            # Bound even the old unbounded reader while reproducing the failure.
            timeout=min(kwargs.get("timeout", 2), 2),
        )
        self.probe_results.append(result)
        return {
            "output": result.stdout.decode("utf-8", errors="replace"),
            "returncode": result.returncode,
        }


@pytest.fixture
def install_environment(monkeypatch):
    def install(env, *, env_type="local"):
        monkeypatch.setenv("_HERMES_GATEWAY", "1")
        monkeypatch.setattr(terminal, "_active_environments", {"default": env})
        monkeypatch.setattr(terminal, "_last_activity", {"default": 0.0})
        monkeypatch.setattr(terminal, "_task_env_overrides", {})
        monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(terminal, "get_session_cwd", lambda *_: None)
        monkeypatch.setattr(terminal, "record_session_cwd", lambda *_: None)
        monkeypatch.setattr(
            terminal,
            "_get_env_config",
            lambda: {
                "env_type": env_type,
                "cwd": env.cwd,
                "timeout": 60,
                "lifetime_seconds": 3600,
            },
        )
        monkeypatch.setattr(
            terminal, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True}
        )
        return env

    return install


@pytest.mark.parametrize("oversized", [False, True])
def test_local_binary_is_skipped_without_fallback(
    tmp_path, install_environment, oversized
):
    executable = tmp_path / "node"
    payload = b"\x7fELF\x00\nhermes gateway restart\n"
    if oversized:
        payload += b"#" * _MAX_REFERENCED_SCRIPT_BYTES
    executable.write_bytes(payload)
    command = f"{shlex.quote(str(executable))} --version"
    env = install_environment(RecordingEnvironment(tmp_path, command))

    result = json.loads(terminal.terminal_tool(command=command))

    assert result["exit_code"] == 0
    assert [call[0] for call in env.calls] == [command]


def test_missing_local_script_does_not_probe_environment(tmp_path, install_environment):
    command = f"bash {shlex.quote(str(tmp_path / 'missing.sh'))}"
    env = install_environment(RecordingEnvironment(tmp_path, command))

    result = json.loads(terminal.terminal_tool(command=command))

    assert result["exit_code"] == 0
    assert [call[0] for call in env.calls] == [command]


@pytest.mark.parametrize(
    "kind", ["unsafe-script", "oversized-text", "directory", "fifo"]
)
def test_local_unsafe_reference_is_blocked_without_execution(
    tmp_path, install_environment, kind
):
    if kind in {"directory", "fifo"} and os.name != "posix":
        pytest.skip("nonregular-file descriptor behavior requires POSIX")
    script = tmp_path / "script.sh"
    if kind == "unsafe-script":
        script.write_text("#!/bin/sh\nhermes gateway restart\n")
    elif kind == "oversized-text":
        script.write_bytes(b"#" * (_MAX_REFERENCED_SCRIPT_BYTES + 1))
    elif kind == "directory":
        script.mkdir()
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("named pipes require POSIX")
        os.mkfifo(script)
    command = f"bash {shlex.quote(str(script))}"
    env = install_environment(RecordingEnvironment(tmp_path, command))

    result = json.loads(terminal.terminal_tool(command=command))

    assert result["exit_code"] == 1
    assert "referenced script" in result["error"]
    assert env.calls == []


@pytest.mark.skipif(os.name != "posix", reason="remote probes use a POSIX shell")
@pytest.mark.parametrize(
    ("payload", "blocked"),
    [
        (b"#!/bin/sh\nprintf 'healthy\\n'\n", False),
        (b"#!/bin/sh\nhermes gateway restart\n", True),
        (b"\x7fELF\x00\nhermes gateway restart\n", False),
        (b"\x7fELF\x00" + b"#" * (_MAX_REFERENCED_SCRIPT_BYTES + 100), False),
        (b"#" * _MAX_REFERENCED_SCRIPT_BYTES, False),
        (b"#" * (_MAX_REFERENCED_SCRIPT_BYTES + 100), True),
        (("#" + "é" * (_MAX_REFERENCED_SCRIPT_BYTES // 2 + 100)).encode(), True),
    ],
    ids=[
        "safe-script",
        "unsafe-script",
        "binary",
        "large-binary",
        "at-size-limit",
        "oversized-text",
        "oversized-utf8",
    ],
)
def test_remote_fallback_is_bounded_and_preserves_lifecycle_protection(
    tmp_path, install_environment, payload, blocked
):
    script = tmp_path / "remote script.sh"
    script.write_bytes(payload)
    remote_path = str(tmp_path / "remote-only" / "remote script.sh")
    command = f"bash {shlex.quote(remote_path)}"
    env = install_environment(
        RecordingEnvironment(
            tmp_path, command, remote_path=remote_path, remote_file=script
        ),
        env_type="ssh",
    )

    result = json.loads(terminal.terminal_tool(command=command))

    assert result["exit_code"] == (1 if blocked else 0)
    probes = [call for call in env.calls if call[0] != command]
    assert len(probes) == 1
    assert 0 < probes[0][1].get("timeout", 0) <= 10
    assert len(env.probe_results[0].stdout) <= _MAX_REFERENCED_SCRIPT_BYTES + 1
    assert (command in [call[0] for call in env.calls]) is not blocked


@pytest.mark.skipif(os.name != "posix", reason="remote probes use a POSIX shell")
def test_remote_fifo_probe_returns_without_reading_pipe(tmp_path, install_environment):
    if not hasattr(os, "mkfifo"):
        pytest.skip("named pipes require POSIX")
    fifo = tmp_path / "script.fifo"
    os.mkfifo(fifo)
    remote_path = str(tmp_path / "remote-only" / "script.fifo")
    command = f"bash {shlex.quote(remote_path)}"
    env = install_environment(
        RecordingEnvironment(
            tmp_path, command, remote_path=remote_path, remote_file=fifo
        ),
        env_type="ssh",
    )

    result = json.loads(terminal.terminal_tool(command=command))

    assert result["exit_code"] == 0
    assert len(env.probe_results) == 1
    assert env.probe_results[0].returncode != 0
    assert env.probe_results[0].stdout == b""
