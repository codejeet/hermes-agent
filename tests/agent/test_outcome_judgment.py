"""Delivery judgment must reach every agent without depending on tools or SOUL."""

from unittest.mock import patch

import pytest

from agent.system_prompt import build_system_prompt_parts
from tests.agent.test_system_prompt import _make_agent, _prompt_parts
from tools.delegate_tool import _build_child_system_prompt


@pytest.mark.parametrize('platform', ['cli', 'discord', 'cron', 'subagent'])
def test_outcome_judgment_reaches_isolated_and_toolless_agents(platform):
    agent = _make_agent(platform=platform, skip_context_files=True, valid_tool_names=[])
    parts = _prompt_parts(agent)
    assert '# Outcome-first judgment' in parts['stable']
    assert 'Achieve the user\'s intended result' in parts['stable']
    assert 'positive sales bias' in parts['stable']
    assert 'seventh-grade' in parts['stable']
    assert 'Passing tests' in parts['stable']
    assert 'learn the general principle' in parts['stable']
    assert '# Outcome-first judgment' not in parts['volatile']


def test_judgment_is_stable_across_rebuild_and_not_duplicated_for_kanban():
    agent = _make_agent(skip_context_files=True, _kanban_worker_guidance='# Kanban task execution protocol')
    first = _prompt_parts(agent)
    second = _prompt_parts(agent)
    assert first['stable'] == second['stable']
    assert first['stable'].count('# Outcome-first judgment') == 1


def test_soul_identity_does_not_replace_shared_judgment():
    agent = _make_agent(load_soul_identity=True, skip_context_files=True)
    with patch('run_agent.load_soul_md', return_value='CUSTOM IDENTITY'):
        parts = build_system_prompt_parts(agent)
    assert parts['stable'].startswith('CUSTOM IDENTITY')
    assert '# Outcome-first judgment' in parts['stable']


def test_delegated_artifact_is_not_replaced_with_work_report():
    prompt = _build_child_system_prompt('Write the landing page hero', context='Audience: spa owner. Goal: free trial.')
    assert 'return the requested deliverable itself' in prompt
    assert 'Keep internal research limits and implementation notes separate' in prompt
    assert 'Audience: spa owner. Goal: free trial.' in prompt


def test_cached_prompt_upgrade_preserves_history_prefix_and_persists_once():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent.outcome_judgment import ensure_cached_outcome_guidance, OUTCOME_FIRST_GUIDANCE

    db = MagicMock()
    old = 'Original identity and instructions\n\nOriginal context'
    agent = SimpleNamespace(_cached_system_prompt=old, _cached_system_prompt_static='Original identity',
                            _session_db=db, session_id='existing-session')
    assert ensure_cached_outcome_guidance(agent) is True
    assert agent._cached_system_prompt == old + '\n\n' + OUTCOME_FIRST_GUIDANCE.strip()
    assert agent._cached_system_prompt_static == 'Original identity'
    db.update_system_prompt.assert_called_once_with('existing-session', agent._cached_system_prompt)
    assert ensure_cached_outcome_guidance(agent) is False
    assert db.update_system_prompt.call_count == 1


def test_new_prompt_needs_no_upgrade():
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent.outcome_judgment import ensure_cached_outcome_guidance, OUTCOME_FIRST_GUIDANCE

    db = MagicMock()
    agent = SimpleNamespace(_cached_system_prompt='Identity\n\n' + OUTCOME_FIRST_GUIDANCE,
                            _session_db=db, session_id='new-session')
    assert ensure_cached_outcome_guidance(agent) is False
    db.update_system_prompt.assert_not_called()


def test_cache_write_failure_does_not_drop_instruction_or_break_turn(caplog):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from agent.outcome_judgment import ensure_cached_outcome_guidance

    db = MagicMock()
    db.update_system_prompt.side_effect = OSError('disk write failed')
    agent = SimpleNamespace(_cached_system_prompt='Old prompt', _session_db=db, session_id='existing')
    assert ensure_cached_outcome_guidance(agent) is True
    assert '# Outcome-first judgment' in agent._cached_system_prompt
    assert 'Could not persist outcome guidance' in caplog.text
