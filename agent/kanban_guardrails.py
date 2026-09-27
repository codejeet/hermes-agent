"""Kanban worker guardrails for proportional implementation and handoff."""

from __future__ import annotations

from typing import Any, Mapping, Optional


KANBAN_ANTI_OVERCOMPLICATION_GUIDANCE = """# Anti-overcomplication check

You must do this after implementation and the smallest relevant test: inspect the
final diff and the actual product behavior once more. Simplify before you complete
the card.

Remove anything the task did not need: speculative abstractions, extra phases or
cards, review/release ceremony, exact-evidence or provenance machinery, acceptance
matrices, defensive UX, approval gates, security theatre, and invented privacy or
compliance constraints. In particular, do not add PII masking, HIPAA ceremony,
consent gates, or generic compliance language to an authenticated internal product
unless the user or an existing concrete requirement calls for it.

Keep a control only when it serves an explicit user requirement, an existing product
contract, a concrete active risk, or an irreversible external action. Ordinary auth,
tenant isolation, credential boundaries, and destructive-action checks are real
controls when the product actually depends on them; do not delete functional safety
just to make the diff smaller. If a questionable-looking control remains, name its
concrete reason in one short note.

Before `kanban_complete` on implementation work, pass:
`anti_overcomplication={"checked": true, "removed": ["..."], "note": "optional"}`.
Use an empty `removed` list when the final pass found nothing extra. This is a final
simplification pass, not a request for another checklist, document, review lane, or
card. Do not create another card for this check.
"""

KANBAN_PRODUCTION_E2E_GUIDANCE = """# Real production E2E

When a card deploys a user-facing feature, E2E must exercise the actual feature
against the real production deployment. A mock, provider-disabled route, DOM-only
assertion, preview environment, or synthetic substitute does not close production E2E.

Use the AJ test client and create or update clearly labeled production test objects as
needed. Those test objects may remain; cleanup is not an acceptance gate. If the
feature calls, texts, or emails, the only authorized real targets are
`+1 818-447-8271` or `aj@scalesystem.ai`. Never contact any other client or lead.
Use one authorized target unless the feature itself requires both channels, and verify
the authoritative application/provider outcome rather than only a click or request.

Before `kanban_complete` on a production-deploy card, pass:
`production_e2e={"checked": true, "environment": "production",
"feature_exercised": "what actually ran", "target": "AJ test client",
"external_action": false}`.
Allowed targets are `AJ test client`, `production public surface`,
`+1 818-447-8271`, and `aj@scalesystem.ai`. Set `external_action=true` only for an
actual call, text, or email to one of the two authorized contact targets.
"""

KANBAN_PLAIN_LANGUAGE_GUIDANCE = """# Human communication

Write for the person reading, not for the system behind the work. This applies to
all front-facing text: interfaces, websites, emails, instructions, explanations,
errors and status updates. Start with what that person needs to understand or do.
Use familiar words, concrete actions and natural sentences. Sound like a helpful
person speaking to another person, not an engineer narrating internal machinery.

Aim for Feynman-like clarity: understand the underlying idea, preserve what makes
it true, and help an unfamiliar reader understand it. Do not merely swap jargon
for shorter words or delete the substance. Build from something familiar, explain
how or why when it matters, and use a concrete example or accurate analogy when
it makes the idea easier to grasp. Keep essential distinctions and causal links;
leave out machinery the reader does not need. Adapt depth to the person's goal,
not to a blanket word limit. The goal is understanding, not just short text.

Do not make readers decode implementation details, abstract process language,
internal classifications, measurement jargon or defensive caveats to get a simple
point. Keep technical detail backstage unless it helps this audience make a
specific decision or they asked for it. When a technical term is necessary,
explain it in ordinary language. Precision does not require jargon; simplicity
does not mean baby talk, vague slogans, clipped fragments or patronizing warmth.

Examples of direction, not fixed replacements or a word blacklist:
- 'Initiate the authentication flow' -> 'Sign in'.
- 'Asset ingestion failed' -> 'We could not upload your file. Try again.'
- 'Review the cohort conversion outcome' -> 'See how many people booked'.
Use a suggested action only when it actually works in that situation.

Be accurate without making the reader carry every internal uncertainty. Keep
necessary limitations clear where they affect the reader's decision; do not
repeat hypothetical warnings or turn an ordinary explanation into a disclaimer.
Do not invent claims or hide material costs, conditions or risks.

Before finishing, read the whole user-facing experience in context, not just a
headline or a few changed strings. Would the intended reader understand it on
first reading, know why it matters and know what to do next? If it sounds like
internal documentation, translate or remove it. Tests passing and checked=true
do not establish this. Include this judgment in the existing simplification pass.
Do not add another attestation, scoring system, checklist artifact or review lane.
Orchestrators and reviewers must carry this principle through briefs and handoffs,
not reintroduce technical alien language as sophistication, precision or safety.
"""

_IMPLEMENTATION_WORKSPACE_KINDS = frozenset({"worktree", "dir"})


def append_kanban_guardrails(base_guidance: str) -> str:
    """Append the worker guardrail once, preserving prompt cache stability."""
    sections = [base_guidance.rstrip()]
    if "# Anti-overcomplication check" not in base_guidance:
        sections.append(KANBAN_ANTI_OVERCOMPLICATION_GUIDANCE.strip())
    if "# Real production E2E" not in base_guidance:
        sections.append(KANBAN_PRODUCTION_E2E_GUIDANCE.strip())
    if "# Human communication" not in base_guidance:
        sections.append(KANBAN_PLAIN_LANGUAGE_GUIDANCE.strip())
    return "\n\n".join(sections)


def implementation_requires_check(task: Any, metadata: Optional[Mapping[str, Any]]) -> bool:
    """Return whether this completion represents implementation work.

    Managed code work normally uses a worktree or a project directory. The
    ``changed_files`` fallback catches implementation performed in a scratch
    workspace without forcing pure research/summarization cards through the gate.
    """
    workspace_kind = str(getattr(task, "workspace_kind", "") or "").strip().lower()
    if workspace_kind in _IMPLEMENTATION_WORKSPACE_KINDS:
        return True
    return bool(isinstance(metadata, Mapping) and metadata.get("changed_files"))


def production_e2e_required(task: Any) -> bool:
    """Return whether the card explicitly includes a production deployment."""
    text = " ".join(
        str(getattr(task, field, "") or "") for field in ("title", "body")
    ).lower()
    if any(marker in text for marker in ("do not deploy", "local-only", "local only")):
        return False
    production = any(
        marker in text
        for marker in ("production", "app.salesystem.ai", "live build", "live deployment")
    )
    deployment = any(
        marker in text
        for marker in ("deploy", "production smoke", "production e2e", "live build")
    )
    return production and deployment


def normalize_anti_overcomplication(value: Any) -> dict[str, Any]:
    """Validate and normalize the deliberately small completion attestation."""
    if not isinstance(value, Mapping):
        raise ValueError(
            "anti_overcomplication must be an object with checked=true and removed=[...]"
        )
    if value.get("checked") is not True:
        raise ValueError("anti_overcomplication.checked must be true after the final pass")
    removed = value.get("removed")
    if not isinstance(removed, list) or any(not isinstance(item, str) for item in removed):
        raise ValueError("anti_overcomplication.removed must be a list of strings (or [])")

    normalized: dict[str, Any] = {
        "checked": True,
        "removed": [item.strip() for item in removed if item.strip()],
    }
    note = value.get("note")
    if note is not None:
        if not isinstance(note, str):
            raise ValueError("anti_overcomplication.note must be a string when provided")
        if note.strip():
            normalized["note"] = note.strip()
    return normalized


_PRODUCTION_E2E_TARGETS = frozenset(
    {
        "AJ test client",
        "production public surface",
        "+1 818-447-8271",
        "aj@scalesystem.ai",
    }
)
_EXTERNAL_E2E_TARGETS = frozenset({"+1 818-447-8271", "aj@scalesystem.ai"})


def normalize_production_e2e(value: Any) -> dict[str, Any]:
    """Validate the live production exercise and its authorized target."""
    if not isinstance(value, Mapping):
        raise ValueError("production_e2e must be an object describing the real production run")
    if value.get("checked") is not True:
        raise ValueError("production_e2e.checked must be true after exercising production")
    if value.get("environment") != "production":
        raise ValueError("production_e2e.environment must be exactly 'production'")

    feature = value.get("feature_exercised")
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError("production_e2e.feature_exercised must name what actually ran")
    target = value.get("target")
    if target not in _PRODUCTION_E2E_TARGETS:
        raise ValueError(
            "production_e2e.target must be AJ test client, production public surface, "
            "+1 818-447-8271, or aj@scalesystem.ai"
        )
    external_action = value.get("external_action")
    if not isinstance(external_action, bool):
        raise ValueError("production_e2e.external_action must be true or false")
    if external_action != (target in _EXTERNAL_E2E_TARGETS):
        raise ValueError(
            "external_action must be true only for the authorized phone or email target"
        )

    return {
        "checked": True,
        "environment": "production",
        "feature_exercised": feature.strip(),
        "target": target,
        "external_action": external_action,
    }
