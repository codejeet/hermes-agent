"""Keep audience-first communication in the actual injected worker guidance."""

from agent.kanban_guardrails import append_kanban_guardrails


def test_worker_guidance_includes_general_human_communication_rules():
    prompt = append_kanban_guardrails("# Kanban task execution protocol")
    for requirement in (
        "# Human communication",
        "Write for the person reading, not for the system behind the work",
        "interfaces, websites, emails, instructions, explanations",
        "Precision does not require jargon",
        "Aim for Feynman-like clarity",
        "The goal is understanding, not just short text",
        "Keep essential distinctions and causal links",
        "whole user-facing experience",
        "Do not add another attestation",
        "Do not invent claims or hide material costs",
    ):
        assert requirement in prompt


def test_communication_guidance_is_added_to_old_guardrails():
    old_prompt = "# Anti-overcomplication check\nOld rules.\n# Real production E2E"
    prompt = append_kanban_guardrails(old_prompt)
    assert "# Human communication" in prompt
    assert prompt.startswith(old_prompt)


def test_communication_guidance_is_idempotent():
    once = append_kanban_guardrails("# Kanban task execution protocol")
    assert "# Human communication" in once
    assert append_kanban_guardrails(once) == once
