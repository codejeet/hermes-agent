"""Context-engine resources must be released during hard agent teardown."""

from unittest.mock import Mock

from run_agent import AIAgent


def test_close_shuts_down_context_engine_once():
    agent = object.__new__(AIAgent)
    engine = Mock()
    agent.context_compressor = engine
    agent._end_session_on_close = False

    agent.close()
    agent.close()

    engine.shutdown.assert_called_once_with()
    assert agent.context_compressor is None


def test_release_clients_shuts_down_context_engine_once():
    """Soft cache eviction discards the agent, so its engine must go too."""
    agent = object.__new__(AIAgent)
    engine = Mock()
    agent.context_compressor = engine

    agent.release_clients()
    agent.release_clients()

    engine.shutdown.assert_called_once_with()
    assert agent.context_compressor is None
