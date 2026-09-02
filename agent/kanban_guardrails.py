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

_IMPLEMENTATION_WORKSPACE_KINDS = frozenset({"worktree", "dir"})


def append_kanban_guardrails(base_guidance: str) -> str:
    """Append the worker guardrail once, preserving prompt cache stability."""
    if "# Anti-overcomplication check" in base_guidance:
        return base_guidance
    return f"{base_guidance.rstrip()}\n\n{KANBAN_ANTI_OVERCOMPLICATION_GUIDANCE.strip()}"


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
