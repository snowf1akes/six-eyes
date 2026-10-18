"""The case worker: a Factory Droid agent that takes each confirmed case to removal or escalation.

main.confirm() starts it when a user confirms a case, main.recheck() starts it after settling a
case past its deadline (so it can tell the user), and POST /cases/{id}/work has it check one case
now. Each start is one fresh Droid session with four tools, all backed by code Six Eyes already has:

    draft_notice(case_id)       the POST /cases/{id}/notice logic: TAKE IT DOWN packets, due in 48 hours
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
TOOL_NAMES = ("draft_notice", "check_live", "escalate", "notify_user")
TURN_TIMEOUT = 300  # seconds for one session; a case needs a handful of tool calls
MAX_MESSAGE = 800   # characters in one update to the user
MAX_MESSAGES = 2    # updates per session
REASONS = {
    "confirmed": "The user just confirmed these posts show them and were shared without their consent.",
    "rechecked": "The 48-hour removal deadline passed and Six Eyes just checked every post on X: removed "
                 "posts are now resolved, posts still up are now escalated, with a complaint to the FTC prepared.",
    "check": "The team asked for a check on this case now.",
}
SYSTEM = """You are the Six Eyes case worker. Six Eyes is the authorized representative of people whose \
intimate images, real or AI-generated, were posted on X without their consent. It asks X to remove them \
under the US TAKE IT DOWN Act, which gives a platform 48 hours from a valid request.

You work one case per session and can act only through your four tools. Every fact you tell the user \
must come from the briefing or a tool result.

- The user just confirmed the case: call draft_notice. If it worked, call notify_user once: the removal \
report for X is ready to submit, X must remove the posts within 48 hours of receiving it, and Six Eyes \
will check back after the deadline.
- Six Eyes just rechecked the case after the deadline: the briefing shows each post's new state. Call \
notify_user once. If posts are still up, say so, tell them the next step is a complaint to the FTC at \
https://takeitdown.ftc.gov, and that the team has the packet and can file it as their authorized \
representative. Pass that URL as link. Don't call check_live or escalate; it's done.
- The team asked for a check: call check_live. If every post is gone, tell the user they were removed. If \
a post is still up after the deadline, call escalate, then notify_user the same way (FTC URL as link). \
Before the deadline, never escalate, and only notify the user if something changed.
- If a tool returns an error, retry it at most once. Tell the user about an error only if they need to act.

Messages to the user: two or three short sentences, calm and plain, no legal jargon. Never describe the \
images, never blame the user, never promise an outcome. In test mode, start the message with "Test:".

End with one sentence for the team log saying what you did."""

log = logging.getLogger("six_eyes.caseworker")
RUNNING: set[tuple[int, int]] = set()  # (user_id, cluster_id) with a session in progress
RUNNING_LOCK = threading.Lock()


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
    """One Droid session on the case: the briefing as the prompt, the four tools, nothing else."""
    config = SessionConfig(system_prompt=SYSTEM, mcp_servers=[create_sdk_mcp_server(SERVER, tools(run))],
                           autonomy=Autonomy.OFF, disable_builtin_skills=True)
    runtime = Runtime(executable=os.environ["DROID_PATH"]) if os.environ.get("DROID_PATH") else None
    with tempfile.TemporaryDirectory() as empty:  # Droid's working directory: no files to read
        async with Session(cwd=empty, model=os.environ.get("CASEWORKER_MODEL", "auto"), config=config,
                           interactions=InteractionHandlers(on_permission=allow_case_tools),
                           runtime=runtime) as session:
            async with session.stream(briefing(run), timeout=TURN_TIMEOUT) as stream:
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
             "Posts in this case:"]
    for case_id, _, url, state, sent_at, due_at in _cluster(run):
        lines.append(f"- case {case_id}: {url}, {state}" + (f", notice made {sent_at}, due {due_at}" if due_at else ""))
    return "\n".join(lines)


def tools(run: Run) -> list:
    """The four tools, bound to this run's case."""
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

    def check(case_id: int) -> ToolResponse:
        return call(check_live, case_id)

    def esc(case_id: int) -> ToolResponse:
        return call(escalate, case_id)

    def notify(user_id: int, text: str, link: str = "") -> ToolResponse:
        return call(notify_user, user_id, text, link)

    return [
        tool("draft_notice", "Make the TAKE IT DOWN removal reports for the case's confirmed posts, one per "
                             "posting account, and start the 48-hour deadline. Safe to call again.", draft),
        tool("check_live", "Check on X whether each of the case's reported posts is still up. Posts that are "
                           "gone are marked removed. Only after the deadline, unless the team asked.", check),
        tool("escalate", "Escalate the case's posts that are still up after the deadline: prepares a "
                         "complaint for the FTC at TakeItDown.ftc.gov. Call check_live first.", esc),
        tool("notify_user", "Text the user a short, plain update about their case (Photon SMS). "
                            "If they need to escalate to the FTC, pass link=https://takeitdown.ftc.gov and tell "
                            "them to open that form; the team has the complaint packet.",
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


def check_live(run: Run, case_id: int) -> str:
    rows = [r for r in _own(run, case_id) if r[3] == "noticed"]
    if run.live is None:
        if not rows:
            raise ToolError("No reported posts waiting on X: nothing to check.")
        run.due = due = min(datetime.fromisoformat(r[5]) for r in rows)
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
    due = min(datetime.fromisoformat(r[5]) for r in up)
    if datetime.now(timezone.utc) < due:
        raise ToolError(f"Not yet: X has until {due.isoformat()} to remove the posts.")
    moved, _ = main.escalate([r[0] for r in up], "caseworker")
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
    if len(text) > MAX_MESSAGE:
        raise ToolError(f"Keep the message under {MAX_MESSAGE} characters.")
    if run.messages >= MAX_MESSAGES:
        raise ToolError("The user already has this session's update.")
    run.messages += 1
    main.tell_user(user_id, run.case_id, text, link or None)
    return "Sent: it's on its way as a text if they enrolled a number."


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
