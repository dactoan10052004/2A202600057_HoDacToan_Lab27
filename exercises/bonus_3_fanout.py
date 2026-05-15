"""Bonus B3 — Multi-reviewer fan-out using LangGraph Send API.

When confidence < ESCALATE_THRESHOLD, the same escalation questions are sent
to TWO reviewers in parallel (two graph branches via Send). Both answers are
collected and merged before synthesizing the final refined review.

Run with:
    .venv/Scripts/python exercises/bonus_3_fanout.py --pr <url>

The graph structure:

    fetch_pr → analyze → route
        └── escalate_fanout
                ├── [Send] → collect_reviewer("A") [interrupt]
                └── [Send] → collect_reviewer("B") [interrupt]
        └── join_reviews → synthesize → commit
    auto_approve / human_approval paths unchanged from exercise 4.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Annotated

import aiosqlite
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent-fanout@v0.1"

SYSTEM_PROMPT = (
    "Senior reviewer. Structured output. "
    "Calibrate your confidence score: >0.73 for trivial PRs only (typo/rename/bump); "
    "0.58-0.73 for small features, schema additions, refactors with some uncertainty; "
    "<0.58 ONLY for clear security vulnerabilities (MD5/SHA1 hashing, SQL injection "
    "via string concat, plaintext token/password storage, hardcoded credentials) or "
    "completely unclear intent. A schema migration with one open question is 0.60-0.70. "
    "If confidence < 0.58, populate escalation_questions with 2-4 specific questions "
    "referencing exact file names and line numbers from the diff."
)


# ─── Extended state to hold two reviewers' answers ─────────────────────────
class FanoutState(ReviewState, total=False):
    # Accumulate answers from both reviewers using a list reducer
    reviewer_answers: Annotated[list[dict], lambda a, b: a + b]
    decision: str
    human_choice: str | None
    human_feedback: str | None


# ─── Shared node implementations ───────────────────────────────────────────
async def _audit(state, entry: AuditEntry) -> None:
    await write_audit_event(
        thread_id=state.get("thread_id", "unknown"),
        pr_url=state.get("pr_url", ""),
        entry=entry,
    )


async def node_fetch_pr(state: FanoutState):
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="fetch_pr", confidence=0.0, risk_level="med",
        decision="pending", reason=f"Fetched {len(pr.files_changed)} files",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"pr_title": pr.title, "pr_diff": pr.diff,
            "pr_files": pr.files_changed, "pr_head_sha": pr.head_sha}


async def node_analyze(state: FanoutState):
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Title: {state['pr_title']}\nDiff:\n{state['pr_diff']}"},
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="analyze", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": a}


async def node_route(state: FanoutState):
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate_fanout"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="route", confidence=c,
        risk_level=risk_level_for(c), decision=decision,
        reason=f"Routing to {decision}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"decision": decision}


# ─── B3 — Fan-out edge mapper (uses Send API) ──────────────────────────────
def _route_or_fanout(state: FanoutState):
    """Conditional edge from route: string for simple paths, Send list for fan-out."""
    decision = state["decision"]
    if decision != "escalate_fanout":
        return decision  # "auto_approve" or "human_approval"
    a = state["analysis"]
    questions = a.escalation_questions or ["What is the intent of this PR?", "Any security concerns?"]
    console.print(f"[cyan]→ escalate_fanout[/cyan]  [dim]sending to 2 reviewers[/dim]")
    payload_base = {
        "pr_url": state["pr_url"],
        "thread_id": state.get("thread_id", ""),
        "pr_diff": state["pr_diff"],
        "pr_title": state["pr_title"],
        "analysis": state["analysis"],
        "reviewer_answers": [],
        "confidence": a.confidence,
        "questions": questions,
        "risk_factors": a.risk_factors,
        "summary": a.summary,
    }
    return [
        Send("collect_reviewer", {**payload_base, "reviewer_label": "Reviewer A"}),
        Send("collect_reviewer", {**payload_base, "reviewer_label": "Reviewer B"}),
    ]


async def node_collect_reviewer(state: FanoutState):
    """Each parallel branch collects one reviewer's answers via interrupt()."""
    label = state.get("reviewer_label", "Reviewer")
    questions = state.get("questions", [])
    console.print(f"[cyan]→ collect_reviewer[/cyan] [{label}]")

    answers = interrupt({
        "kind": "escalation",
        "reviewer_label": label,
        "pr_url": state.get("pr_url"),
        "confidence": state.get("confidence"),
        "summary": state.get("summary"),
        "risk_factors": state.get("risk_factors", []),
        "questions": questions,
    })
    return {"reviewer_answers": [{"label": label, "answers": answers}]}


async def node_join_reviews(state: FanoutState):
    """Merge answers from both reviewers."""
    console.print("[cyan]→ join_reviews[/cyan]")
    all_answers = state.get("reviewer_answers", [])
    console.print(f"  [green]✓[/green] collected {len(all_answers)} reviewer response(s)")
    # Merge: combine answers by question, appending both reviewers' answers
    merged: dict[str, str] = {}
    for entry in all_answers:
        label = entry.get("label", "?")
        for q, a in (entry.get("answers") or {}).items():
            if q in merged:
                merged[q] += f" | [{label}] {a}"
            else:
                merged[q] = f"[{label}] {a}"
    return {"escalation_answers": merged}


async def node_synthesize(state: FanoutState):
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with both reviewers' answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": "Refine review with multi-reviewer answers."},
            {"role": "user", "content": f"Diff:\n{state['pr_diff']}\n\nQ&A:\n{qa}"},
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="synthesize", confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence), decision="pending",
        reason=f"Multi-reviewer synthesis; confidence → {refined.confidence:.0%}",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"analysis": refined}


def _render_comment_body(state) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("escalation_answers"):
        lines.append("\n_Multi-reviewer Q&A:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


async def node_commit(state: FanoutState):
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment")
        action = "committed"
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        action = "commit_failed"
    c = state["analysis"].confidence
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="commit", confidence=c,
        risk_level=risk_level_for(c), decision=action,
        reason="committed after multi-reviewer fan-out",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_auto_approve(state: FanoutState):
    console.print("[cyan]→ auto_approve[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        action = "auto_committed"
    except Exception:
        action = "auto_commit_failed"
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="auto_approve", confidence=a.confidence,
        risk_level=risk_level_for(a.confidence), decision="auto",
        reason=f"High confidence ({a.confidence:.0%})",
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


async def node_human_approval(state: FanoutState):
    a = state["analysis"]
    resp = interrupt({
        "kind": "approval_request", "pr_url": state["pr_url"],
        "confidence": a.confidence, "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary, "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": resp.get("choice"), "human_feedback": resp.get("feedback")}


async def node_human_commit(state: FanoutState):
    console.print("[cyan]→ human_commit[/cyan]")
    t0 = time.monotonic()
    if state.get("human_choice") == "approve":
        try:
            post_review_comment(state["pr_url"], _render_comment_body(state))
            action = "committed"
        except Exception:
            action = "commit_failed"
    else:
        action = "rejected"
    c = state["analysis"].confidence
    await _audit(state, AuditEntry(
        agent_id=AGENT_ID, action="commit", confidence=c, risk_level=risk_level_for(c),
        decision=action, reason=state.get("human_feedback"),
        execution_time_ms=int((time.monotonic() - t0) * 1000),
    ))
    return {"final_action": action}


# ─── Graph wiring ───────────────────────────────────────────────────────────
def build_fanout_graph(checkpointer):
    g = StateGraph(FanoutState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("human_commit", node_human_commit),
        ("collect_reviewer", node_collect_reviewer),
        ("join_reviews", node_join_reviews),
        ("synthesize", node_synthesize), ("commit", node_commit),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    # _route_or_fanout returns a string OR [Send, Send] for fan-out
    g.add_conditional_edges(
        "route", _route_or_fanout,
        {"auto_approve": "auto_approve", "human_approval": "human_approval"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "human_commit")
    g.add_edge("human_commit", END)
    # Fan-out: Send → collect_reviewer × 2 → join_reviews → synthesize → commit
    g.add_edge("collect_reviewer", "join_reviews")
    g.add_edge("join_reviews", "synthesize")
    g.add_edge("synthesize", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


# ─── Interactive CLI handler ────────────────────────────────────────────────
def handle_interrupt(payload):
    kind = payload.get("kind")
    if kind == "approval_request":
        console.print(Panel.fit(payload["summary"],
                                title=f"Approve? conf={payload['confidence']:.0%}",
                                border_style="green"))
        choice = console.input("approve/reject? ").strip().lower()
        return {"choice": choice, "feedback": console.input("Feedback: ").strip()}

    if kind == "escalation":
        label = payload.get("reviewer_label", "Reviewer")
        console.print(Panel.fit(
            payload["summary"],
            title=f"[{label}] Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print(f"  [red]Risks:[/red] {', '.join(payload['risk_factors'])}")
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}

    raise ValueError(f"Unknown interrupt kind: {kind}")


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Bonus B3 — Multi-reviewer fan-out[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with aiosqlite.connect(db_path()) as conn:
        cp = AsyncSqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=[
            ("common.schemas", "PRAnalysis"),
            ("common.schemas", "ReviewComment"),
        ]))
        await cp.setup()
        app = build_fanout_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            if len(interrupts) == 1:
                # Single interrupt (approval or single escalation)
                resume_val = handle_interrupt(interrupts[0].value)
                result = await app.ainvoke(Command(resume=resume_val), cfg)
            else:
                # Multiple parallel interrupts (fan-out reviewers) — resume all at once
                resume_map = {
                    iv.id: handle_interrupt(iv.value)
                    for iv in interrupts
                }
                result = await app.ainvoke(Command(resume=resume_map), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] python -m audit.replay --thread {thread_id}")


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    p.add_argument("--thread", help="Resume existing thread")
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
