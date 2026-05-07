# Voice Fast-Ack Router Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make gateway voice mode feel snappier by speaking a short fast-model acknowledgement immediately after STT, then routing the real job through the normal Hermes agent loop and optionally using a concise spoken summary for final TTS.

**Architecture:** Add a small gateway-local voice fast-ack/router subsystem behind `voice.fast_ack.enabled`. The fast model is a dispatcher/narrator only: it returns strict JSON, may rewrite the job prompt for the primary agent, and never bypasses auth, tools, approvals, or safety. The primary agent still runs through the existing `_run_agent(...)` path.

**Tech Stack:** Python, pytest, `gateway/run.py`, `gateway/platforms/base.py`, existing `agent.auxiliary_client.call_llm` for lightweight model calls, existing `_send_voice_reply` / adapter auto-TTS for speech.

---

### Task 1: Add focused fast-ack unit tests

**Objective:** Capture the core router behavior before implementation.

**Files:**
- Create: `tests/gateway/test_voice_fast_ack_routing.py`

**Step 1: Write failing tests**

Create tests for these behaviors:

1. `test_parse_voice_fast_ack_router_json_strips_fences_and_coerces_fields`
   - Expected helper: `GatewayRunner._parse_voice_fast_ack_router_response(raw)`.
   - Raw fenced JSON should return a dict with string `ack_text`, bool `is_job`, string `job_prompt`, and string `response_mode`.

2. `test_voice_fast_ack_disabled_skips_router_and_tts`
   - Patch `_load_gateway_config` to return `{}`.
   - Call expected helper `await runner._maybe_prepare_voice_fast_ack(event, "hello", source)`.
   - Assert result leaves message unchanged and `_send_voice_reply` is not called.

3. `test_voice_fast_ack_uses_configured_aux_model_and_speaks_ack`
   - Patch `_load_gateway_config` with `voice.fast_ack.enabled=true`, provider/model/api_key/base_url.
   - Patch `gateway.run.call_llm` or the helper import site to return JSON: `{"ack_text":"Got it — checking now.","is_job":true,"job_prompt":"Check the thing","response_mode":"normal"}`.
   - Patch `gateway.run.extract_content_or_reasoning` if needed.
   - Patch `runner._send_voice_reply = AsyncMock(return_value=True)`.
   - Assert `call_llm` received configured provider/model/base_url/api_key, `max_tokens`, timeout derived from `ack_timeout_ms` or `timeout`, and the prompt demands strict JSON.
   - Assert `_send_voice_reply(event, "Got it — checking now.")` called once.
   - Assert prepared message includes the job prompt and preserves the original voice transcript.

4. `test_voice_fast_ack_invalid_json_fails_open_to_original_message`
   - `call_llm` returns non-JSON.
   - Assert no exception, message unchanged, no ack sent.

5. `test_voice_fast_ack_final_summary_sets_spoken_text_for_base_auto_tts`
   - Expected helper: `await runner._maybe_prepare_voice_final_summary(event, long_response)`.
   - Config has `voice.fast_ack.summarize_final=true`.
   - Patch fast summarizer `call_llm` response.
   - Assert `event._hermes_voice_spoken_text` is set to summary and returned text response is unchanged.

6. `test_base_auto_tts_uses_fast_ack_spoken_text_when_present`
   - Patch an adapter subclass / fake that runs the relevant base auto-TTS branch or directly test the extracted helper if available.
   - Assert final TTS uses `event._hermes_voice_spoken_text` instead of long markdown text.

**Step 2: Run tests to verify RED**

Run:

```bash
python -m pytest tests/gateway/test_voice_fast_ack_routing.py -q
```

Expected: FAIL because helpers/imports do not exist yet.

---

### Task 2: Implement parser/config/router helper in `gateway/run.py`

**Objective:** Add the lightweight router helper without touching the main agent loop yet.

**Files:**
- Modify: `gateway/run.py`

**Implementation requirements:**

- Add safe helpers near other `GatewayRunner` helper methods:
  - `_voice_fast_ack_config(self) -> dict`
  - `_voice_fast_ack_enabled(self, event: MessageEvent, cfg: dict) -> bool`
  - `_parse_voice_fast_ack_router_response(raw: str) -> dict | None`
  - `_build_voice_fast_ack_router_messages(transcript: str, cfg: dict) -> list[dict]`
  - `async _maybe_prepare_voice_fast_ack(self, event, message_text, source) -> tuple[str, dict | None]`
- Only run when:
  - `event.message_type == MessageType.VOICE` or `MessageType.AUDIO`
  - `voice.fast_ack.enabled` is truthy
  - message text does not clearly contain STT failure markers.
- Use `agent.auxiliary_client.call_llm` / `extract_content_or_reasoning` inside `asyncio.to_thread`.
- Pass explicit `provider`, `model`, `base_url`, `api_key`; do not log secrets.
- Bound timeout using `ack_timeout_ms / 1000`, falling back to `timeout`, default `1.5` seconds.
- Strict JSON fields:
  - `ack_text`: max short spoken acknowledgement
  - `is_job`: bool
  - `job_prompt`: non-empty only when useful
  - `response_mode`: string, e.g. `normal`, `direct`, `background`, `ask_clarify`
- Fail open on timeout/errors/invalid JSON: return original message and `None`.
- If `ack_text` exists, send with `_send_voice_reply(event, ack_text)`; if that returns false/raises, fall back to adapter text send with thread metadata.
- If `job_prompt` exists, produce:

```text
[Voice router rewrote this into a clearer job prompt.
Original voice transcript: <original>]

<job_prompt>
```

**Step 2: Run tests to verify GREEN for parser/router**

Run:

```bash
python -m pytest tests/gateway/test_voice_fast_ack_routing.py -q
```

Expected: parser/router tests pass; final summary/base TTS tests may still fail if not implemented yet.

---

### Task 3: Wire fast ack into the live gateway voice path

**Objective:** Invoke the router after STT and before the heavy agent turn.

**Files:**
- Modify: `gateway/run.py`

**Implementation requirements:**

- In `_handle_message_with_agent(...)`, immediately after:

```python
message_text = await self._prepare_inbound_message_text(...)
if message_text is None:
    return
```

call:

```python
message_text, _voice_fast_ack_route = await self._maybe_prepare_voice_fast_ack(
    event,
    message_text,
    source,
)
```

- Store `_voice_fast_ack_route` locally for final summary decisions.
- Do not move auth/session-running guards.
- Do not alter `_run_agent(...)` safety/approval/tool behavior.

**Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/gateway/test_voice_fast_ack_routing.py -q
python -m pytest tests/gateway/test_voice_command.py -q
```

Expected: all pass.

---

### Task 4: Add final spoken summary support

**Objective:** Let the fast model turn a long/markdown-heavy final response into a natural spoken summary for voice TTS while preserving the full text response.

**Files:**
- Modify: `gateway/run.py`
- Modify: `gateway/platforms/base.py`

**Implementation requirements:**

- Add `async _maybe_prepare_voice_final_summary(self, event, response, route=None) -> str | None`.
- Only run when:
  - voice/audio event
  - `voice.fast_ack.summarize_final` truthy
  - response is non-empty
  - response is long or markdown-heavy, or route response_mode asks for summary.
- Use same fast-ack configured provider/model and `call_llm` with strict natural speech prompt.
- Bound output by `max_spoken_summary_seconds`; cap text length defensively.
- Set `event._hermes_voice_spoken_text = summary` in `_handle_message_with_agent(...)` before final return.
- In `gateway/platforms/base.py` auto-TTS branch, use:

```python
speech_text = getattr(event, "_hermes_voice_spoken_text", None) or text_content
```

instead of always passing `text_content` to TTS.
- Avoid duplicate final TTS. Keep existing `_should_send_voice_reply` behavior unless streaming `already_sent` requires runner-side speech.

**Step 2: Run targeted tests**

Run:

```bash
python -m pytest tests/gateway/test_voice_fast_ack_routing.py -q
python -m pytest tests/gateway/test_voice_command.py -q
```

Expected: all pass.

---

### Task 5: Integration review and docs/skill update

**Objective:** Verify the feature is safe, tested, and documented for future Hermes voice work.

**Files:**
- Modify if needed: `skills/autonomous-ai-agents/hermes-agent/references/voice-fast-ack-model-routing.md` or equivalent installed skill if repo skill is not the active copy.

**Verification commands:**

```bash
python -m pytest tests/gateway/test_voice_fast_ack_routing.py -q
python -m pytest tests/gateway/test_voice_command.py -q
python -m pytest tests/gateway/test_session_model_override_routing.py -q
python -m pytest tests/gateway/test_config.py -q
python -m py_compile gateway/run.py gateway/platforms/base.py
```

**Acceptance criteria:**

- Fast ack only activates for voice/audio when `voice.fast_ack.enabled=true`.
- Fast ack uses configured provider/model rather than the session primary model or `/model` override.
- Invalid/slow fast model output never blocks the normal agent turn.
- Router rewrites preserve original transcript.
- Final summary changes spoken TTS only, not the text result.
- Existing voice command and model override tests still pass.
---

### Addendum: Filtered thought fast speech progress narration

**Goal:** In addition to the initial ack and final summary, Spark can narrate safe mid-turn progress from the observable tool transcript.

**Config:**

```yaml
voice:
  fast_ack:
    filtered_thought_fast_speech:
      enabled: true
      min_interval_seconds: 10
      max_updates_per_turn: 3
      max_words: 18
      max_tokens: 120
      ack_timeout_ms: 2500
      emotion: curious, focused, warm
      include_visible_thoughts: false
```

**Behavior:**

- `_run_agent(...)` now accepts the original `MessageEvent` and, for voice/audio fast-ack turns, keeps the agent `tool_progress_callback` active even when visible text tool progress is disabled.
- Tool lifecycle events are queued into a separate async speech narrator as sanitized activity categories, not raw tool arguments/previews.
- Spark receives a compact sanitized activity transcript and returns strict JSON: `should_speak`, `speech_text`.
- The prompt and local post-filter explicitly forbid revealing private chain-of-thought, hidden reasoning, raw prompts, secrets, file contents, raw tool arguments, and raw tool outputs. It only narrates high-level observable progress.
- Speech updates use `_send_voice_reply(...)`; if Spark or TTS fails, the feature fails open and the main agent turn continues.
- `/voice off` disables this path through the same fast-ack gate.
