"""Keep approved real calls possible without authorizing customer test calls."""
import pytest
from agent.kanban_guardrails import append_kanban_guardrails, normalize_production_e2e


def receipt(target, external_action=True):
    return dict(checked=True, environment="production", feature_exercised="Live inbound PSTN call", target=target, external_action=external_action)


def test_runtime_prompt_preserves_task_specific_owned_number_authority():
    prompt = append_kanban_guardrails("# Worker")
    assert "Real handset calls are allowed" in prompt
    assert "AJ-authorized owned Twilio numbers" in prompt
    assert "Verify ownership" in prompt
    assert "Never contact any other client or lead" in prompt
    assert "the only authorized real targets are" not in prompt


def test_owned_number_call_can_be_reported_as_real_external_action():
    result = normalize_production_e2e(receipt("AJ-authorized owned Twilio numbers"))
    assert result["external_action"] is True


def test_owned_number_call_cannot_be_reported_as_no_external_action():
    with pytest.raises(ValueError, match="external_action"):
        normalize_production_e2e(receipt("AJ-authorized owned Twilio numbers", False))


@pytest.mark.parametrize("target", ["customer@example.com", "+15555550123", "any owned number"])
def test_unapproved_targets_still_rejected(target):
    with pytest.raises(ValueError, match="target"):
        normalize_production_e2e(receipt(target))


def test_real_aj_handset_call_remains_allowed():
    assert normalize_production_e2e(receipt("+1 818-447-8271"))["external_action"] is True
