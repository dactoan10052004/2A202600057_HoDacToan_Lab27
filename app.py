"""Exercise 5 + Bonus B1 (Time-travel) + B2 (Calibration) — Streamlit UI.

Run with:
    .venv/Scripts/streamlit run app.py

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     success card — reviewer does nothing
    58 – 72%     human_approval   Approve / Reject / Edit buttons
    <  58%       escalate         question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── DB helpers ────────────────────────────────────────────────────────────
async def _fetch_recent_threads(limit: int = 10) -> list[dict]:
    try:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id, pr_url,
                       MIN(timestamp) AS started,
                       MAX(timestamp) AS last_event,
                       MAX(risk_level) AS worst_risk,
                       COUNT(*) AS events
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT ?
                """,
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


async def _fetch_calibration() -> dict:
    """B2 — query audit_events for confidence calibration stats."""
    try:
        async with db_conn() as conn:
            # Overall per-decision averages
            async with conn.execute(
                """
                SELECT decision, COUNT(*) AS cnt, AVG(confidence) AS avg_conf
                  FROM audit_events
                 WHERE action IN ('human_approval','auto_approve','commit')
                   AND decision NOT IN ('pending','escalate')
                 GROUP BY decision
                 ORDER BY avg_conf DESC
                """
            ) as cur:
                by_decision = [dict(r) for r in await cur.fetchall()]

            # Overall totals
            async with conn.execute(
                """
                SELECT COUNT(*) AS total,
                       AVG(confidence) AS avg_conf,
                       SUM(CASE WHEN decision='approve' THEN 1 ELSE 0 END) AS approved,
                       SUM(CASE WHEN decision='reject'  THEN 1 ELSE 0 END) AS rejected,
                       SUM(CASE WHEN decision='auto'    THEN 1 ELSE 0 END) AS auto_approved
                  FROM audit_events
                 WHERE action IN ('human_approval','auto_approve')
                   AND decision NOT IN ('pending','escalate')
                """
            ) as cur:
                totals = dict(await cur.fetchone())

        return {"by_decision": by_decision, "totals": totals}
    except Exception:
        return {}


async def _resume_from_checkpoint(thread_id: str, checkpoint_id: str):
    """B1 — resume graph from a specific earlier checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        return await app.ainvoke(None, cfg)


async def _fetch_checkpoints(thread_id: str) -> list[dict]:
    """B1 — list unique LangGraph checkpoints for a thread.

    Deduplicates by (step, next_node) — keeps the most recent checkpoint
    for each unique position in the graph, so repeated resume cycles don't
    clutter the list.
    """
    try:
        async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
            await cp.setup()
            cfg = {"configurable": {"thread_id": thread_id}}
            app = build_graph(cp)
            seen: set[tuple] = set()
            history = []
            async for state in app.aget_state_history(cfg):
                metadata = state.metadata or {}
                step = metadata.get("step", "?")
                next_nodes = tuple(sorted(state.next))
                key = (step, next_nodes)
                if key in seen:
                    continue
                seen.add(key)
                history.append({
                    "checkpoint_id": (state.config.get("configurable") or {}).get("checkpoint_id", ""),
                    "step": step,
                    "next": list(state.next),
                    "ts": state.created_at,
                })
            return history
    except Exception:
        return []


# ─── Session state ─────────────────────────────────────────────────────────
for key, default in [
    ("thread_id", None), ("pr_url", ""),
    ("interrupt_payload", None), ("final", None),
    ("timetravel_thread", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Recent sessions
    st.header("Recent sessions")
    try:
        threads = asyncio.run(_fetch_recent_threads(limit=8))
        if threads:
            for t in threads:
                risk_icon = {"low": "🟢", "med": "🟡", "high": "🔴"}.get(t.get("worst_risk", ""), "⚪")
                label = f"{risk_icon} `{t['thread_id'][:8]}…`"
                if st.button(label, key=f"sess_{t['thread_id']}", help=t.get("pr_url", "")):
                    st.session_state.thread_id = t["thread_id"]
                    st.session_state.pr_url = t.get("pr_url", "")
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No sessions yet.")
    except Exception:
        st.caption("(audit_events not initialised)")

    st.divider()

    # B2 — Confidence calibration
    st.header("📊 Calibration (B2)")
    try:
        cal = asyncio.run(_fetch_calibration())
        totals = cal.get("totals", {})
        if totals and totals.get("total"):
            col_a, col_b = st.columns(2)
            col_a.metric("Total reviews", int(totals["total"]))
            avg = totals.get("avg_conf") or 0
            col_b.metric("Avg confidence", f"{avg:.0%}")
            st.caption(
                f"Auto-approved: {int(totals.get('auto_approved') or 0)}  "
                f"· Approved: {int(totals.get('approved') or 0)}  "
                f"· Rejected: {int(totals.get('rejected') or 0)}"
            )
            for row in cal.get("by_decision", []):
                st.progress(
                    float(row["avg_conf"]),
                    text=f"{row['decision']} — avg {row['avg_conf']:.0%} ({int(row['cnt'])} reviews)",
                )
            # Over/under confidence check
            if totals.get("avg_conf") and totals["avg_conf"] > 0.75:
                st.warning("⚠️ Model may be over-confident (avg > 75%)")
            elif totals.get("avg_conf") and totals["avg_conf"] < 0.55:
                st.info("ℹ️ Model may be under-confident (avg < 55%)")
        else:
            st.caption("Run some reviews first.")
    except Exception:
        st.caption("(stats unavailable)")

    st.divider()

    # B1 — Time-travel
    st.header("⏪ Time-travel (B1)")
    tt_thread = st.text_input(
        "Thread ID to inspect",
        value=st.session_state.thread_id or "",
        key="tt_input",
        placeholder="paste thread_id here",
    )
    if st.button("Load checkpoints") and tt_thread:
        st.session_state.timetravel_thread = tt_thread

    if st.session_state.timetravel_thread:
        with st.spinner("Loading checkpoints..."):
            checkpoints = asyncio.run(_fetch_checkpoints(st.session_state.timetravel_thread))
        if checkpoints:
            st.caption(f"{len(checkpoints)} checkpoint(s) found")
            for i, ck in enumerate(checkpoints):
                next_nodes = ", ".join(ck["next"]) if ck["next"] else "END"
                label = f"Step {ck['step']} → {next_nodes}"
                if st.button(label, key=f"ck_{i}", help=f"checkpoint_id: {ck['checkpoint_id']}"):
                    # Resume from this checkpoint with a fresh invocation
                    st.session_state.thread_id = st.session_state.timetravel_thread
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    with st.spinner(f"Resuming from step {ck['step']}..."):
                        result = asyncio.run(_resume_from_checkpoint(
                            st.session_state.timetravel_thread,
                            ck["checkpoint_id"],
                        ))
                    if "__interrupt__" in result:
                        st.session_state.interrupt_payload = result["__interrupt__"][0].value
                    else:
                        st.session_state.final = result
                    st.rerun()
        else:
            st.caption("No checkpoints found for this thread.")


# ─── Renderers ─────────────────────────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_input("Feedback (required for Edit)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit (auto-rewrite)"):
        if not feedback:
            st.warning("Provide feedback before clicking Edit.")
        else:
            return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])
    with st.form("escalation"):
        answers: dict[str, str] = {}
        for i, question in enumerate(payload.get("questions", [])):
            answers[question] = st.text_input(question, key=f"esc_q_{i}")
        if st.form_submit_button("Submit answers"):
            return answers
    return None


# ─── Graph helpers ─────────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


# ─── Main flow ─────────────────────────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")

if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(
        f"thread_id = `{st.session_state.thread_id}`  ·  "
        f"replay: `python -m audit.replay --thread {st.session_state.thread_id}`"
    )
