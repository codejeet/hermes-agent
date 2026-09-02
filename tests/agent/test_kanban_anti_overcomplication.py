"""Worker prompt guardrails against ceremony and security theatre."""


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


def test_guardrail_is_appended_to_worker_guidance_once():
    from agent.kanban_guardrails import append_kanban_guardrails

    base = "# Kanban task execution protocol"
    once = append_kanban_guardrails(base)
    twice = append_kanban_guardrails(once)

    assert once.startswith(base)
    assert once.count("# Anti-overcomplication check") == 1
    assert twice == once
