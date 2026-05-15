# Lab 27 — HITL PR Review Agent: Individual Report

**Student:** Ho Dac Toan  
**Student ID:** 2A202600057  
**Date:** 2026-05-15  
**Repository:** `2A202600057_HoDacToan_Lab27`  

---

## 1. Overview

This lab builds a Human-in-the-Loop (HITL) pull-request review agent using **LangGraph**, **Streamlit**, and **SQLite**. The agent reads a GitHub PR diff, has an LLM analyze it with a self-reported confidence score, then routes to one of three branches:

| Confidence | Branch | Behavior |
|---|---|---|
| > 73% | `auto_approve` | Agent posts the review comment directly — no human needed |
| 58–73% | `human_approval` | Agent pauses, shows reviewer the diff + LLM reasoning + Approve/Reject/Edit buttons |
| < 58% | `escalate` | Agent asks the reviewer specific LLM-generated questions, re-synthesizes the review with answers, then commits |

Every node writes one row to a structured `audit_events` SQLite table. Sessions can be replayed or resumed from any checkpoint via LangGraph's `AsyncSqliteSaver`.

---

## 2. Architecture

```
fetch_pr → analyze → route
                       ├── auto_approve → [commit comment]
                       ├── human_approval (interrupt) → commit
                       └── escalate (interrupt) → synthesize → commit
```

All graph state is persisted in `hitl_audit.db` via `AsyncSqliteSaver`. The same file also stores the `audit_events` table, keeping the setup to a single file with no external dependencies.

**Key design decisions:**
- `JsonPlusSerializer(allowed_msgpack_modules=[("common.schemas", "PRAnalysis"), ...])` is passed to `AsyncSqliteSaver` to eliminate deserialization warnings when reading Pydantic objects from checkpoints.
- `interrupt()` is called inside nodes; `Command(resume=value)` is used to continue — this is the standard LangGraph HITL pattern.
- The Streamlit app drives the same graph as the CLI, sharing `exercise_4_audit.py`'s `build_graph()`.

---

## 3. Exercise Implementations

### Exercise 1 — Confidence Routing (`exercise_1_confidence.py`)

Implemented `node_analyze` (calls LLM with `with_structured_output(PRAnalysis)`), `node_route` (reads `state["analysis"].confidence` and returns the routing key), and full graph wiring with `add_conditional_edges`.

The system prompt was calibrated to prevent LLM overconfidence:
```
> 0.73  for trivial PRs only (typo/rename/bump)
0.58–0.73  for small features, schema additions, refactors
< 0.58  ONLY for clear security vulnerabilities (MD5, SQL injection, hardcoded credentials)
```

**Result:** PR #1 routes to `human_approval` (68%), PR #2 routes to `escalate` (55%) — different branches as required.

---

### Exercise 2 — HITL with `interrupt()` (`exercise_2_hitl.py`)

Implemented `node_human_approval` which calls `interrupt()` with the full approval payload (confidence, summary, comments, diff preview). In `main()`, added the resume loop:

```python
while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    answer = prompt_human(payload)
    result = app.invoke(Command(resume=answer), cfg)
```

Compiled the graph with `MemorySaver` checkpointer — required for `interrupt()` to work (without a checkpointer, `interrupt()` raises `GraphInterrupt` immediately).

---

### Exercise 3 — Escalation with Q&A (`exercise_3_escalation.py`)

Implemented two new nodes:

- **`node_escalate`**: calls `interrupt({"kind": "escalation", "questions": [...], ...})` with the LLM-generated `escalation_questions` from `PRAnalysis`. If the LLM didn't generate questions, falls back to generic ones.
- **`node_synthesize`**: re-prompts the LLM with the original diff + reviewer's Q&A answers. Returns a refined `PRAnalysis` with updated confidence.

Added edges: `escalate → synthesize → commit`.

**Observed behaviour on PR #2:** LLM generated 4 specific questions referencing exact file/line numbers (`auth.py:7` for MD5, `storage.py:22` for SQL injection). After reviewer answers, confidence rose from 55% → 75%.

---

### Exercise 4 — SQLite Audit Trail (`exercise_4_audit.py`)

Implemented the `audit()` helper and added one `AuditEntry` to every node:

| Node | action | decision |
|---|---|---|
| `node_fetch_pr` | `fetch_pr` | `pending` |
| `node_analyze` | `analyze` | `pending` |
| `node_route` | `route` | routing key |
| `node_auto_approve` | `auto_approve` | `auto` |
| `node_human_approval` | `human_approval` | `pending` → `approve`/`reject`/`edit` |
| `node_commit` | `commit` | `committed`/`rejected`/`committed_after_edit` |
| `node_escalate` | `escalate` | `escalate` → `pending` |
| `node_synthesize` | `synthesize` | `pending` |
| `node_auto_commit` | `auto_approve` | `auto` |
| `node_human_commit` | `commit` | final action |

Switched from `AsyncSqliteSaver.from_conn_string()` (no `serde` support in v3.1.0) to direct `aiosqlite.connect()` + `AsyncSqliteSaver(conn, serde=JsonPlusSerializer(...))`.

**Audit replay — PR #1 (human_approval path):**
```
fetch_pr   conf=0.00  decision=pending   2390ms  Fetched 3 files
analyze    conf=0.68  decision=pending  12155ms  new feature with some potential risk
route      conf=0.68  decision=human_approval   Routing to human_approval
human_approval  conf=0.68  decision=approve  reviewer=dactoan12345  Looks good
commit     conf=0.68  decision=committed  1344ms
```

**Audit replay — PR #2 (escalation path):**
```
fetch_pr   conf=0.00  decision=pending   2500ms  Fetched 4 files
analyze    conf=0.55  decision=pending   8061ms  significant security concerns
route      conf=0.55  decision=escalate  routing to escalate
escalate   conf=0.55  decision=pending   reviewer=dactoan12345  4 questions answered
synthesize conf=0.75  decision=pending   6828ms  confidence 55% → 75%
commit     conf=0.75  decision=committed 1405ms
```

---

### Exercise 5 — Streamlit Approval UI (`app.py`)

Implemented the full Streamlit web UI:

**`run_graph(pr_url, thread_id, resume_value=None)`** — wraps `AsyncSqliteSaver` + `build_graph()`. First call uses `ainvoke({"pr_url": ..., "thread_id": ...})`. Resume calls use `ainvoke(Command(resume=resume_value))`.

**`render_approval_card(payload)`** — shows confidence, LLM reasoning, comment list, diff preview, and three buttons: **Approve** / **Reject** / **Edit (auto-rewrite)**. Returns `{"choice": ..., "feedback": ...}` or `None` if no button was clicked yet.

**`render_escalation_card(payload)`** — renders a Streamlit form with one `st.text_input` per LLM-generated question. Submits all answers at once via `st.form_submit_button`.

`thread_id` is stored in `st.session_state` so the session persists across reruns. The `interrupt_payload` and `final` state are also in session state to handle Streamlit's rerun model.

---

## 4. Bonus Challenges

### B1 — Time-Travel (sidebar in `app.py`)

Used `app.aget_state_history(config)` to list all checkpoints for a thread. Deduplicated by `(step, next_nodes)` tuple to prevent repeated resume cycles from cluttering the list. Each checkpoint button calls `_resume_from_checkpoint(thread_id, checkpoint_id)` which re-invokes the graph from that exact state.

```python
async for state in app.aget_state_history(cfg):
    key = (step, tuple(sorted(state.next)))
    if key in seen: continue
    seen.add(key)
    history.append({...})
```

### B2 — Confidence Calibration (sidebar in `app.py`)

Queries `audit_events` to compute per-decision average confidence and overall totals:

```sql
SELECT decision, COUNT(*), AVG(confidence)
FROM audit_events
WHERE action IN ('human_approval','auto_approve','commit')
  AND decision NOT IN ('pending','escalate')
GROUP BY decision
```

Displays as `st.progress` bars per decision type. Flags over-confidence (avg > 75%) and under-confidence (avg < 55%) with warnings.

### B3 — Multi-Reviewer Fan-out (`exercises/bonus_3_fanout.py`)

Uses the LangGraph `Send` API to fan out escalation to two reviewers in parallel. The key fix: `Send` objects must be returned from a **conditional edge function**, not from a node body:

```python
def _route_or_fanout(state):
    if state["decision"] != "escalate_fanout":
        return state["decision"]  # simple string routing
    return [
        Send("collect_reviewer", {**payload_base, "reviewer_label": "Reviewer A"}),
        Send("collect_reviewer", {**payload_base, "reviewer_label": "Reviewer B"}),
    ]
```

Multiple parallel interrupts are resumed using `Command(resume={iv.id: answer for iv in interrupts})` — a dict mapping each interrupt's `id` to its answer.

`node_join_reviews` merges both reviewers' answers by question:
```
[Reviewer A] JWT token auth | [Reviewer B] JWT token auth 2
```
`node_synthesize` then produces a unified refined review.

**Result on PR #2:** 55% → fan-out → both reviewers answered → join → synthesize → 70% → `committed`.

### B4 — Auto-Edit (`exercise_4_audit.py`, `node_commit`)

When the reviewer clicks **Edit** and provides feedback, the agent re-prompts the LLM to rewrite the review using that feedback before posting:

```python
elif state.get("human_choice") == "edit" and state.get("human_feedback"):
    llm = get_llm().with_structured_output(PRAnalysis)
    refined_analysis = await llm.ainvoke([
        {"role": "system", "content": "Rewrite the PR review incorporating the reviewer's feedback."},
        {"role": "user", "content": f"Original review:\n{...}\n\nReviewer feedback: {feedback}"},
    ])
    # post refined_analysis instead of original
    action = "committed_after_edit"
```

The **Edit** button in `render_approval_card()` requires non-empty feedback before submitting, preventing accidental edits.

---

## 5. Key Technical Challenges

**1. `AsyncSqliteSaver.from_conn_string()` does not accept `serde` parameter (v3.1.0)**  
→ Fixed by using `aiosqlite.connect(db_path())` + `AsyncSqliteSaver(conn, serde=JsonPlusSerializer(...))` directly.

**2. `Deserializing unregistered type PRAnalysis` warning**  
→ `JsonPlusSerializer` uses msgpack internally and warns on unregistered Pydantic models. Fixed by passing `allowed_msgpack_modules=[("common.schemas", "PRAnalysis"), ("common.schemas", "ReviewComment")]` — this explicitly whitelists the types and suppresses the warning.

**3. B3 `InvalidUpdateError: Expected dict, got [Send(...)]`**  
→ LangGraph nodes must return `dict`. Only conditional edge functions can return `Send` objects. Moved fan-out logic from `node_escalate_fanout` (a node) into `_route_or_fanout` (the edge mapper function).

**4. B3 Multiple parallel interrupts**  
→ When two `collect_reviewer` branches both interrupt, `Command(resume=value)` fails with "must specify interrupt id". Fixed by using `Command(resume={iv.id: handle_interrupt(iv.value) for iv in interrupts})`.

**5. LLM overconfidence (always auto-approve at 85%)**  
→ Fixed with a detailed calibration system prompt listing exact confidence tiers with examples. PR #1 now reliably routes to `human_approval` (~68%) and PR #2 to `escalate` (~55%).

---

## 6. Test Evidence

All exercises tested with real API calls against `https://github.com/VinUni-AI20k/PR-Demo`.

| Test | PR | Path | Result |
|---|---|---|---|
| Exercise 1 | PR#1 | `human_approval` | ✅ confidence=68%, different branch from PR#2 |
| Exercise 1 | PR#2 | `escalate` | ✅ confidence=55% |
| Exercise 4 CLI | PR#1 | approve → commit | ✅ `committed`, comment posted to GitHub |
| Exercise 4 CLI | PR#2 | escalate → synthesize | ✅ 55%→75%, `committed` |
| Exercise 5 (Streamlit) | PR#1 | human_approval UI | ✅ Approve button → committed |
| Exercise 5 (Streamlit) | PR#2 | escalation form UI | ✅ 4 Q&A → submitted → committed |
| Bonus B1 | PR#1 | time-travel | ✅ checkpoints listed, resume from earlier step works |
| Bonus B2 | all | calibration sidebar | ✅ avg confidence displayed per decision |
| Bonus B3 | PR#2 | fan-out 2 reviewers | ✅ parallel interrupts, join, synthesize → committed |
| Bonus B4 | PR#1 | edit → auto-rewrite | ✅ LLM rewrites review using feedback |

14 sessions recorded in `audit_events` table (visible in Streamlit sidebar and via `python -m audit.replay --list`).

---

## 7. Running the Project

```bash
# Setup
git clone <repo>
cd 2A202600057_HoDacToan_Lab27
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in OPENAI_API_KEY and GITHUB_TOKEN

# Set environment (Windows PowerShell)
$env:PYTHONPATH = "d:\path\to\2A202600057_HoDacToan_Lab27"
$env:PYTHONUTF8 = "1"

# Run exercises (CLI)
.venv/Scripts/python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
.venv/Scripts/python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1

# Run Streamlit UI (Exercise 5 + B1 + B2)
.venv/Scripts/streamlit run app.py

# Run B3 fan-out
.venv/Scripts/python exercises/bonus_3_fanout.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2

# Replay audit session
.venv/Scripts/python -m audit.replay --list
.venv/Scripts/python -m audit.replay --thread <thread_id>
```
