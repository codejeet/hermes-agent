"""Worker prompt guardrails against ceremony and security theatre."""

from types import SimpleNamespace

import pytest


def test_guardrail_prompt_requires_a_post_implementation_simplification_pass():
    from agent.kanban_guardrails import KANBAN_ANTI_OVERCOMPLICATION_GUIDANCE

    guidance = KANBAN_ANTI_OVERCOMPLICATION_GUIDANCE
    assert "Anti-overcomplication check" in guidance
    assert "after implementation" in guidance
    assert "security theatre" in guidance
    assert "HIPAA" in guidance
    assert "exact-evidence" in guidance
    assert "smallest relevant test" in guidance
    assert "Do not create another card" in guidance


def test_guardrail_requires_real_production_e2e_for_deployed_features():
    from agent.kanban_guardrails import KANBAN_PRODUCTION_E2E_GUIDANCE

    guidance = KANBAN_PRODUCTION_E2E_GUIDANCE
    assert "actual feature" in guidance
    assert "real production" in guidance
    assert "+1 818-447-8271" in guidance
    assert "aj@scalesystem.ai" in guidance
    assert "AJ test client" in guidance
    assert "test objects may remain" in guidance
    assert "Never contact any other client" in guidance


def test_guardrail_is_appended_to_worker_guidance_once():
    from agent.kanban_guardrails import append_kanban_guardrails

    base = "# Kanban task execution protocol"
    once = append_kanban_guardrails(base)
    twice = append_kanban_guardrails(once)

    assert once.startswith(base)
    assert once.count("# Anti-overcomplication check") == 1
    assert once.count("# Real production E2E") == 1
    assert twice == once


def test_production_e2e_gate_only_applies_to_explicit_live_deploys():
    from agent.kanban_guardrails import production_e2e_required

    assert production_e2e_required(
        SimpleNamespace(title="Ship client action", body="Deploy to production and smoke live.")
    )
    assert not production_e2e_required(
        SimpleNamespace(title="Refactor parser", body="Run focused unit tests locally.")
    )
    assert not production_e2e_required(
        SimpleNamespace(title="Document deployment", body="Do not deploy to production.")
    )


def test_production_e2e_rejects_non_aj_external_targets():
    from agent.kanban_guardrails import normalize_production_e2e

    with pytest.raises(ValueError, match="target"):
        normalize_production_e2e(
            {
                "checked": True,
                "environment": "production",
                "feature_exercised": "Sent the live message",
                "target": "customer@example.com",
                "external_action": True,
            }
        )
