"""Shared shell state must not persist per-call delegation/claim identity."""
import json
import shlex
import sys
from pathlib import Path

import pytest

from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    delegated_child_context,
)
from tools.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX bash snapshot")


def probe(env):
    code = (
        "import json,os; print(json.dumps({k:v for k,v in os.environ.items() "
        "if k == 'HERMES_DELEGATED_CHILD_CONTEXT' or k.startswith('HERMES_KANBAN_')}))"
    )
    result = env.execute(f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}")
    assert result["returncode"] == 0, result
    return json.loads(result["output"].strip())


@pytest.fixture
def shell(monkeypatch, tmp_path):
    for key in (DELEGATED_CHILD_ENV_MARKER, *KANBAN_ENV_KEYS):
        monkeypatch.delenv(key, raising=False)
    env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    assert env._snapshot_ready
    try:
        yield env
    finally:
        env.cleanup()


def test_child_command_does_not_poison_later_parent(shell):
    with delegated_child_context():
        assert probe(shell)[DELEGATED_CHILD_ENV_MARKER] == "1"
    assert DELEGATED_CHILD_ENV_MARKER not in probe(shell)
    assert DELEGATED_CHILD_ENV_MARKER not in Path(shell._snapshot_path).read_text()


def test_parent_claim_does_not_leak_back_into_child(shell, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_parent")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "123")
    assert probe(shell)["HERMES_KANBAN_TASK"] == "t_parent"
    with delegated_child_context():
        values = probe(shell)
    assert values == {DELEGATED_CHILD_ENV_MARKER: "1"}
    assert probe(shell)["HERMES_KANBAN_TASK"] == "t_parent"


def test_existing_poisoned_snapshot_uses_current_call_identity(shell):
    with open(shell._snapshot_path, "a") as stream:
        stream.write('declare -x HERMES_DELEGATED_CHILD_CONTEXT="1"\n')
        stream.write('declare -x HERMES_KANBAN_TASK="t_foreign"\n')
    assert probe(shell) == {}
    with delegated_child_context():
        assert probe(shell) == {DELEGATED_CHILD_ENV_MARKER: "1"}


def test_child_bootstrap_excludes_lineage_from_snapshot(monkeypatch, tmp_path):
    monkeypatch.delenv(DELEGATED_CHILD_ENV_MARKER, raising=False)
    with delegated_child_context():
        env = LocalEnvironment(cwd=str(tmp_path), timeout=15)
    try:
        assert env._snapshot_ready
        assert DELEGATED_CHILD_ENV_MARKER not in Path(env._snapshot_path).read_text()
        assert DELEGATED_CHILD_ENV_MARKER not in probe(env)
    finally:
        env.cleanup()
