"""The case worker: a Factory Droid agent that takes each confirmed case to removal or escalation.

main.confirm() starts it when a user confirms a case. It drafts the notice, texts the user, then
owns the 48-hour clock: wait_until either sleeps until due_at (short demo deadlines) or arms a
timer that starts a fresh session at the deadline. Dashboard load, POST /recheck and Modal's hourly
sweep only wake overdue sessions if that timer died with the container. POST /timewarp/{id} cuts
the wait short. POST /cases/{id}/work has it check one case now.

Each start is one fresh Droid session with five tools, all backed by code Six Eyes already has:

    draft_notice(case_id)       the POST /cases/{id}/notice logic: TAKE IT DOWN packets, due in 48 hours
    wait_until(case_id)         sleep until due_at when that's short; otherwise schedule a deadline session
    check_live(case_id)         main.post_status (x.is_live) on every noticed post in the clone cluster;
                                removed posts become resolved (main.resolve)
    escalate(case_id)           posts still up after the deadline become escalated with an FTC complaint
                                packet (main.escalate)
    notify_user(user_id, text)  a short text to the user (Photon SMS)

The agent decides what to do and how to tell the user; the tools hold the rules, so it can't act
outside this case's cluster, escalate before the deadline or without seeing a post still up, or
make more than one billed X check per session. Droid's own tools (shell, files, web) are refused.

Needs the droid CLI on PATH (or DROID_PATH) and FACTORY_API_KEY, or a logged-in droid.
CASEWORKER=0 turns it off; CASEWORKER_MODEL picks the model (default "auto", Factory's router).
"""
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from droid_sdk import (Autonomy, InteractionHandlers, McpToolAction, PermissionRequest, PermissionResponse,
                       Runtime, Session, SessionConfig, ToolConfirmationOutcome)
from droid_sdk.mcp import ToolResponse, create_sdk_mcp_server, tool
from fastapi import HTTPException

import channels
import main

SERVER = "six-eyes"
TOOL_NAMES = ("draft_notice", "wait_until", "check_live", "escalate", "notify_user")
TURN_TIMEOUT = 300  # seconds of model/tool work in one session, not counting wait_until
WAIT_INLINE_MAX = 120  # seconds we will sleep inside a turn (NOTICE_DEADLINE_SECONDS=5 demos)
MAX_MESSAGE = 800   # characters in one update to the user
MAX_MESSAGES = 2    # updates per session
REASONS = {
    "confirmed": "The user just confirmed these posts show them and were shared without their consent.",
    "deadline": "The 48-hour removal deadline just passed. Check every noticed post on X, resolve "
                "removed ones, escalate any still up, and tell the user.",
    "check": "The team asked for a check on this case now.",
}
SYSTEM = """You are the Six Eyes case worker. Six Eyes is the authorized representative of people whose \
intimate images, real or AI-generated, were posted on X without their consent. It asks X to remove them \
under the US TAKE IT DOWN Act, which gives a platform 48 hours from a valid request.

You work one case per session and can act only through your five tools. Every fact you tell the user \
must come from the briefing or a tool result.

- The user just confirmed the case: call draft_notice, then wait_until. Six Eyes already texted them \
that the report is ready; do not notify_user after draft_notice. If wait_until says the deadline has \
passed (or the wait was skipped), call check_live; if a post is still up after the deadline, call \
escalate, then notify_user with https://takeitdown.ftc.gov as link. Put that FTC URL and the dashboard \
URL from the briefing in the text. If wait_until says a new session will start later, stop.
- The removal deadline just passed: call check_live. If every post is gone, tell the user they were \
removed. If a post is still up, call escalate, then notify_user the same way (both URLs). \
Don't call wait_until.
- The team asked for a check: call check_live. If every post is gone, tell the user they were removed. If \
a post is still up after the deadline, call escalate, then notify_user the same way (both URLs). \
Before the deadline, never escalate, and only notify the user if something changed.
- If a tool returns an error, retry it at most once. Tell the user about an error only if they need to act.

Messages to the user: two or three short sentences, calm and plain, no legal jargon. Never describe the \
images, never blame the user, never promise an outcome. In test mode, start the message with "Test:".

End with one sentence for the team log saying what you did."""

log = logging.getLogger("six_eyes.caseworker")
RUNNING: set[tuple[int, int]] = set()  # (user_id, cluster_id) with a session in progress
RUNNING_LOCK = threading.Lock()
TIMERS: dict[tuple[int, int], threading.Timer] = {}  # scheduled deadline session
WAKE: dict[tuple[int, int], threading.Event] = {}    # in-session wait_until, so time-warp can cut it short
TIMERS_LOCK = threading.Lock()


class ToolError(Exception):
    """Shown to the agent as a failed tool call, so it can change course."""


@dataclass
class Run:
    case_id: int
    user_id: int
    cluster_id: int
    reason: str
    live: dict[str, bool] | None = None  # post url -> still up, from this session's check_live
    due: datetime | None = None          # the notice's deadline, as check_live saw it
    messages: int = 0
    calls: list[str] = field(default_factory=list)


def enabled() -> bool:
    return os.environ.get("CASEWORKER", "1") != "0" and bool(os.environ.get("DROID_PATH") or shutil.which("droid"))


def load(case_id: int, reason: str) -> Run | None:
    with closing(main.db()) as con:
        row = con.execute("SELECT user_id, cluster_id FROM cases WHERE id = ?", (case_id,)).fetchone()
    return Run(case_id, row[0], row[1] if row[1] is not None else case_id, reason) if row else None


def start(case_id: int, reason: str) -> bool:
    """Work the case in a background thread; False if the case worker is off or already on this cluster."""
    if reason not in REASONS or not enabled() or not (run := load(case_id, reason)):
        return False
    key = (run.user_id, run.cluster_id)
    with RUNNING_LOCK:
        if key in RUNNING:
            return False
        RUNNING.add(key)
    threading.Thread(target=_work_and_record, args=(run, key), daemon=True).start()
    return True


def _work_and_record(run: Run, key: tuple[int, int]) -> None:
    try:
        detail = asyncio.run(work(run))
    except Exception as e:  # the case stays as it is, logged below

        log.exception("case worker on case %s failed", run.case_id)
        detail = {"outcome": "error", "error": f"{type(e).__name__}: {e}", "tools": run.calls}
    finally:
        with RUNNING_LOCK:
            RUNNING.discard(key)
    _event([run.case_id], "caseworker", {"reason": run.reason, **detail})


async def work(run: Run) -> dict:
    """One Droid session on the case: the briefing as the prompt, the five tools, nothing else."""
    config = SessionConfig(system_prompt=SYSTEM, mcp_servers=[create_sdk_mcp_server(SERVER, tools(run))],
                           autonomy=Autonomy.OFF, disable_builtin_skills=True)
    runtime = Runtime(executable=os.environ["DROID_PATH"]) if os.environ.get("DROID_PATH") else None
    key = os.environ.get("FACTORY_API_KEY") or None
    with tempfile.TemporaryDirectory() as empty:  # Droid's working directory: no files to read
        async with Session(cwd=empty, model=os.environ.get("CASEWORKER_MODEL", "auto"), config=config,
                           interactions=InteractionHandlers(on_permission=allow_case_tools),
                           runtime=runtime, api_key=key) as session:
            async with session.stream(briefing(run), timeout=TURN_TIMEOUT + WAIT_INLINE_MAX) as stream:
                async for _ in stream:
                    pass
            result = stream.result
    return {"outcome": result.subtype, "session_id": result.session_id, "summary": result.text, "tools": run.calls}


def allow_case_tools(request: PermissionRequest) -> PermissionResponse:
    """Approve calls to the case tools and refuse everything else Droid might try."""
    def ours(a) -> bool:
        name = a.actual_tool_name or a.tool_name
        return a.server_name == SERVER if a.server_name else name.endswith(TOOL_NAMES)
    allowed = bool(request.actions) and all(isinstance(a, McpToolAction) and ours(a) for a in request.actions)
    return request.respond(ToolConfirmationOutcome.PROCEED_ONCE if allowed else ToolConfirmationOutcome.CANCEL)


def briefing(run: Run) -> str:
    with closing(main.db()) as con:
        name = con.execute("SELECT name FROM users WHERE id = ?", (run.user_id,)).fetchone()[0]
    lines = [REASONS[run.reason], f"Now: {_now()}",
             f"Case {run.case_id}, for user {run.user_id} ({name}). Test mode: {'yes' if main.dry_run() else 'no'}.",
             f"Dashboard: {main.dashboard_url(run.user_id)}",
             f"FTC form: {channels.escalation()['ftc']}",
             "Posts in this case:"]
    for case_id, _, url, state, sent_at, due_at in _cluster(run):
        lines.append(f"- case {case_id}: {url}, {state}" + (f", notice made {sent_at}, due {due_at}" if due_at else ""))
    return "\n".join(lines)


def tools(run: Run) -> list:
    """The five tools, bound to this run's case."""
    def call(f, *args) -> ToolResponse:
        run.calls.append(f"{f.__name__}({', '.join(map(repr, args))})")
        try:
            return ToolResponse(content=f(run, *args))
        except ToolError as e:
            return ToolResponse(content=str(e), is_error=True)
        except Exception:
            log.exception("case tool %s failed", f.__name__)
            return ToolResponse(content="Something went wrong on Six Eyes' side.", is_error=True)

    def draft(case_id: int) -> ToolResponse:
        return call(draft_notice, case_id)

    def wait(case_id: int) -> ToolResponse:
        return call(wait_until, case_id)

    def check(case_id: int) -> ToolResponse:
        return call(check_live, case_id)

    def esc(case_id: int) -> ToolResponse:
        return call(escalate, case_id)

    def notify(user_id: int, text: str, link: str = "") -> ToolResponse:
        return call(notify_user, user_id, text, link)

    return [
        tool("draft_notice", "Make the TAKE IT DOWN removal reports for the case's confirmed posts, one per "
                             "posting account, and start the 48-hour deadline. Safe to call again.", draft),
        tool("wait_until", "Wait until this case's removal deadline. Sleeps when the deadline is soon "
                           "(demo). Otherwise schedules a new session at due_at and ends this one.", wait),
        tool("check_live", "Check on X whether each of the case's reported posts is still up. Posts that are "
                           "gone are marked removed. Only after the deadline, unless the team asked.", check),
        tool("escalate", "Escalate the case's posts that are still up after the deadline: prepares a "
                         "complaint for the FTC at TakeItDown.ftc.gov. Call check_live first.", esc),
        tool("notify_user", "Text the user a short, plain update about their case (Photon SMS). "
                            "If they need to escalate to the FTC, pass link=https://takeitdown.ftc.gov and include "
                            "that URL and the dashboard URL from the briefing in the text.",
             notify),
    ]


def draft_notice(run: Run, case_id: int) -> str:
    rows = _own(run, case_id)
    if confirmed := [r for r in rows if r[3] == "confirmed"]:
        try:
            n = main.notice(confirmed[0][0])
        except HTTPException as e:
            raise ToolError(f"The reports weren't made: {e.detail}")
        accounts = ", ".join(f"@{p['reported_username']}" for p in n["packets"])
        return (f"Made {len(n['packets'])} removal report(s) for {n['platform']} ({accounts}) covering "
                f"{len(n['case_ids'])} post(s). Someone submits them on {n['form_url']}; the deadline is "
                f"{n['due_at']}." + (" Test mode: they won't be submitted." if n["dry_run"] else ""))
    if noticed := [r for r in rows if r[3] == "noticed"]:
        return (f"The reports were already made on {noticed[0][4]} for {len(noticed)} post(s); "
                f"the deadline is {noticed[0][5]}.")
    return f"Nothing to draft: the posts are {', '.join(sorted({r[3] for r in rows}))}."


def wait_until(run: Run, case_id: int) -> str:
    """Sleep until due_at when that's short; otherwise arm a deadline session and stop this one."""
    rows = [r for r in _own(run, case_id) if r[3] == "noticed" and r[5]]
    if not rows:
        raise ToolError("No deadline to wait for: call draft_notice first.")
    due = _as_utc(min(r[5] for r in rows))
    remaining = (due - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return f"The deadline already passed at {due.isoformat()}. Check the posts now."
    if remaining > WAIT_INLINE_MAX:
        _arm(run.case_id, due)
        return (f"The deadline is {due.isoformat()}, too far to wait in this session. "
                f"A new session will start then. Tell the user we'll check back, then stop.")
    ev = threading.Event()
    key = (run.user_id, run.cluster_id)
    with TIMERS_LOCK:
        WAKE[key] = ev
    try:
        skipped = ev.wait(timeout=remaining)
    finally:
        with TIMERS_LOCK:
            WAKE.pop(key, None)
    if skipped:
        return "The wait was skipped (test). The deadline is now; check the posts now."
    return f"Waited until {due.isoformat()}. The deadline has passed; check the posts now."


def check_live(run: Run, case_id: int) -> str:
    rows = [r for r in _own(run, case_id) if r[3] == "noticed"]
    if run.live is None:
        if not rows:
            raise ToolError("No reported posts waiting on X: nothing to check.")
        run.due = due = min(_as_utc(r[5]) for r in rows)
        if datetime.now(timezone.utc) < due and run.reason != "check":
            raise ToolError(f"X has until {due.isoformat()} to remove the posts; check after that.")
        try:
            run.live = main.post_status([(post_id, url) for _, post_id, url, *_ in rows])
        except RuntimeError as e:
            raise ToolError(str(e))
        except httpx.HTTPError:
            raise ToolError("Couldn't reach X to check the posts. Try again later.")
        main.resolve([r[0] for r in rows if not run.live[r[2]]], "caseworker")
    up = [url for url, is_up in run.live.items() if is_up]
    removed = [url for url, is_up in run.live.items() if not is_up]
    past = datetime.now(timezone.utc) >= run.due
    return "\n".join([f"Deadline {run.due.isoformat()} ({'passed' if past else 'not passed yet'})."]
                     + [f"Still up: {u}" for u in up] + [f"Removed: {u}" for u in removed])


def escalate(run: Run, case_id: int) -> str:
    rows = _own(run, case_id)
    if run.live is None:
        raise ToolError("Call check_live first: only posts seen still up can be escalated.")
    up = [r for r in rows if r[3] == "noticed" and run.live.get(r[2])]
    if not up:
        raise ToolError("Nothing to escalate: no reported post is still up.")
    due = min(_as_utc(r[5]) for r in up)
    if datetime.now(timezone.utc) < due:
        raise ToolError(f"Not yet: X has until {due.isoformat()} to remove the posts.")
    moved, *_ = main.escalate([r[0] for r in up], "caseworker")
    if not moved:
        raise ToolError("Already escalated.")
    return (f"Escalated {len({r[2] for r in up if r[0] in moved})} post(s) still up after the {due.isoformat()} "
            f"deadline, with an FTC complaint packet for the team to file at {channels.escalation()['ftc']}."
            + (" Test mode: it won't be filed." if main.dry_run() else ""))


def notify_user(run: Run, user_id: int, text: str, link: str = "") -> str:
    if user_id != run.user_id:
        raise ToolError(f"This case belongs to user {run.user_id}; only they can be notified.")
    text = " ".join(text.split())
    if not text:
        raise ToolError("The message is empty.")
    ftc = channels.escalation()["ftc"]
    dash = main.dashboard_url(user_id)
    if (link or "").rstrip("/") == ftc.rstrip("/") and dash not in text:
        extra = f" FTC: {ftc}  Guide: {dash}"
        if len(text) + 1 + len(extra) <= MAX_MESSAGE:
            text = f"{text} {extra}"
    if len(text) > MAX_MESSAGE:
        raise ToolError(f"Keep the message under {MAX_MESSAGE} characters.")
    if run.messages >= MAX_MESSAGES:
        raise ToolError("The user already has this session's update.")
    run.messages += 1
    main.tell_user(user_id, run.case_id, text, link or None)
    return "Sent: it's on its way as a text if they enrolled a number."


def wake_due(user_id: int | None = None) -> list[int]:
    """Start a deadline session for each overdue noticed cluster. Used when the in-process timer died."""
    with closing(main.db()) as con:
        due = con.execute(
            "SELECT user_id, COALESCE(cluster_id, id) AS k, MIN(id) FROM cases WHERE state = 'noticed' AND due_at <= ?"
            + (" AND user_id = ?" if user_id is not None else "") + " GROUP BY user_id, k ORDER BY 3",
            (_now(),) if user_id is None else (_now(), user_id)).fetchall()
    started = []
    for uid, cluster, first in due:
        _disarm(uid, cluster)
        if start(first, "deadline"):
            started.append(first)
    return started


def skip_deadline(case_id: int) -> bool:
    """Cut the wait short (time-warp). Wakes an in-session wait or starts a deadline session."""
    run = load(case_id, "deadline")
    if not run:
        return False
    key = (run.user_id, run.cluster_id)
    with TIMERS_LOCK:
        ev = WAKE.get(key)
        timer = TIMERS.pop(key, None)
    if timer:
        timer.cancel()
    if ev is not None:
        ev.set()
        return True
    with RUNNING_LOCK:
        if key in RUNNING:
            return True
    return start(case_id, "deadline")


def wait_settled(case_ids: list[int], timeout: float | None = None) -> dict | None:
    """Block until none of the cases are still noticed, then return the dashboard's settled shape."""
    if not case_ids:
        return None
    limit = time.time() + (TURN_TIMEOUT + WAIT_INLINE_MAX if timeout is None else timeout)
    marks = ",".join("?" * len(case_ids))
    while time.time() < limit:
        with closing(main.db()) as con:
            rows = con.execute(f"SELECT id, state FROM cases WHERE id IN ({marks})", case_ids).fetchall()
        if rows and all(state != "noticed" for _, state in rows):
            resolved = [i for i, state in rows if state == "resolved"]
            escalated = [i for i, state in rows if state == "escalated"]
            packet, ftc, form, dry = None, None, None, None
            if escalated:
                with closing(main.db()) as con:
                    detail = con.execute("SELECT detail FROM events WHERE case_id = ? AND kind = 'escalated' "
                                         "ORDER BY rowid DESC", (escalated[0],)).fetchone()
                body = json.loads(detail[0]) if detail else {}
                packet, ftc = body.get("ftc_packet"), body.get("ftc_url") or channels.escalation()["ftc"]
                form, dry = body.get("ftc_form"), body.get("dry_run")
            return {"resolved": resolved, "escalated": escalated, "ftc_packet": packet, "ftc_url": ftc,
                    "ftc_form": form, "dry_run": dry}
        time.sleep(0.05)
    return None


def restore() -> None:
    """On API start: fire overdue clusters, arm timers for noticed cases still waiting."""
    if not enabled():
        return
    now = datetime.now(timezone.utc)
    with closing(main.db()) as con:
        rows = con.execute(
            "SELECT user_id, COALESCE(cluster_id, id) AS k, MIN(id), MIN(due_at) FROM cases "
            "WHERE state = 'noticed' AND due_at IS NOT NULL GROUP BY user_id, k").fetchall()
    for uid, cluster, first, due_at in rows:
        due = _as_utc(due_at)
        if due <= now:
            start(first, "deadline")
        else:
            _arm(first, due)
            log.info("armed case worker for cluster %s at %s (user %s)", cluster, due.isoformat(), uid)


def _arm(case_id: int, due: datetime) -> None:
    run = load(case_id, "deadline")
    if not run:
        return
    key = (run.user_id, run.cluster_id)
    delay = max(0.0, (due - datetime.now(timezone.utc)).total_seconds())
    with TIMERS_LOCK:
        old = TIMERS.pop(key, None)
        if old:
            old.cancel()
        timer = threading.Timer(delay, _fire, args=(run.case_id, key))
        timer.daemon = True
        TIMERS[key] = timer
        timer.start()


def _fire(case_id: int, key: tuple[int, int]) -> None:
    with TIMERS_LOCK:
        TIMERS.pop(key, None)
    start(case_id, "deadline")


def _disarm(user_id: int, cluster_id: int) -> None:
    with TIMERS_LOCK:
        timer = TIMERS.pop((user_id, cluster_id), None)
    if timer:
        timer.cancel()


def _as_utc(iso: str) -> datetime:
    due = datetime.fromisoformat(iso)
    return due if due.tzinfo else due.replace(tzinfo=timezone.utc)


def _cluster(run: Run) -> list[tuple]:
    """(id, post_id, url, state, sent_at, due_at) for every case in the run's clone cluster."""
    with closing(main.db()) as con:
        return con.execute("SELECT id, post_id, url, state, sent_at, due_at FROM cases "
                           "WHERE user_id = ? AND (id = ? OR cluster_id = ?) ORDER BY id",
                           (run.user_id, run.cluster_id, run.cluster_id)).fetchall()


def _own(run: Run, case_id: int) -> list[tuple]:
    rows = _cluster(run)
    if case_id not in {r[0] for r in rows}:
        raise ToolError(f"Case {case_id} isn't part of this case; use case {run.case_id}.")
    return rows


def _event(case_ids: list[int], kind: str, detail: dict) -> None:
    at = _now()
    with closing(main.db()) as con, con:
        con.executemany("INSERT INTO events (case_id, at, kind, detail) VALUES (?, ?, ?, ?)",
                        [(i, at, kind, json.dumps(detail)) for i in case_ids])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
