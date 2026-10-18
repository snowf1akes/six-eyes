# six eyes
legal agent that automates the process of identifying and taking down deep fakes

## After the 48 hours

The TAKE IT DOWN Act gives a platform 48 hours to remove a reported intimate depiction and its
identical copies. The **Factory case worker** owns that clock. After it drafts the notice it calls
`wait_until`: if the deadline is soon (a local demo with `NOTICE_DEADLINE_SECONDS=5`) it sleeps
inside the same Droid session, then checks X. If the deadline is farther away it arms an in-process
timer that starts a fresh session at `due_at`.

Modal still stops the API container after 10 idle minutes, which would kill that timer. Opening the
dashboard (`GET /cases/{user_id}`) or `POST /recheck` (Modal's hourly `deadline_sweep`) only **wakes
overdue Factory sessions** if the timer died. If the case worker is off, those paths settle the
cluster themselves:

- each post in the cluster is checked on X (`x.is_live`)
- posts that are gone become **resolved**
- posts still up become **escalated**, with an FTC complaint packet stored on the case

`POST /timewarp/{case_id}` is test-mode only: it sets `due_at` to now and cuts the Factory wait
short, so a demo doesn't wait two days. The dashboard's "Skip the 48-hour wait (test)" button
calls it.

Nothing is filed with the FTC automatically. `escalate` fills
[`api/templates/ftc_packet.md`](api/templates/ftc_packet.md) with the still-up URLs, PDQ hashes, the
cluster's timeline, the evidence scores and the notice copy, plus the
[TakeItDown.ftc.gov](https://takeitdown.ftc.gov) link, and stores that packet in an `escalated`
event. A person submits it. The user is **texted** through Photon (`api/photon.py`) if they enrolled
a phone number: X missed the 48 hours, escalate at TakeItDown.ftc.gov, the team has the packet.

## The case worker (Factory Droid)

Finding a deepfake is half the job; the other half is following it until it's gone. The case worker
is a [Factory Droid](https://docs.factory.ai/sdk/python) agent that takes each confirmed case from
"the user says this is them, shared without consent" to either "removed" or "escalated", and keeps
the user told along the way. It lives in [`api/caseworker.py`](api/caseworker.py).

### When it runs

| Trigger | Where | What the agent is asked to do |
| --- | --- | --- |
| The user confirms a case ("This is me" + "Not consensual") | `POST /cases/{id}/confirm` | Make the reports, tell the user, then `wait_until` the deadline and close the case |
| The deadline timer fires, or a missed wake (dashboard / `/recheck` / time-warp) | `wait_until` or `wake_due` | Check X, resolve or escalate, tell the user |
| The team asks for a check | `POST /cases/{id}/work` | Check now, even before the deadline (never escalates early) |

Each run is one fresh Droid session (`droid_sdk.Session`) in a background thread, briefed with the
case's posts, states and deadline. A cluster has at most one session at a time. If the case worker
is off, `recheck` still texts the user itself.

### Its five tools

The tools are Python functions served to Droid over an in-process MCP server
(`droid_sdk.mcp.create_sdk_mcp_server`). They call code Six Eyes already has:

| Tool | Calls | Effect |
| --- | --- | --- |
| `draft_notice(case_id)` | `main.notice()`, the `POST /cases/{id}/notice` logic | TAKE IT DOWN reports for every post in the clone cluster, one per posting account; the cluster moves to `noticed`, due in 48 hours. Safe to repeat. |
| `wait_until(case_id)` | in-process sleep or `threading.Timer` | Sleeps when `due_at` is soon; otherwise schedules a deadline session. Time-warp cuts it short. |
| `check_live(case_id)` | `main.post_status` (`x.is_live`) on every noticed post | Posts that are gone move to `resolved`. |
| `escalate(case_id)` | `main.escalate` | Posts still up after the deadline move to `escalated`, with an FTC complaint packet for TakeItDown.ftc.gov. |
| `notify_user(user_id, text)` | Photon SMS via `main.tell_user` | A short text if they enrolled a number. |

### What the agent decides, and what it can't

The agent chooses which tools to call and writes every message to the user: plain, calm, two or three
sentences, never describing the images, never promising an outcome. The rules live in the tools, so
a wrong turn by the model fails as a tool error it can recover from instead of doing harm:

- It can only act on the case it was started for, and only message that case's user.
- `check_live` makes at most one billed X call per post per session, and only after the deadline
  (unless the team asked).
- `escalate` needs a `check_live` in the same session that saw the post still up, and a passed deadline.
- At most two messages per session, each under 800 characters.
- Droid's built-in tools are refused: `Autonomy.OFF` sends every action to a permission handler that
  approves only these five tools, the session runs in an empty temporary folder, and built-in skills
  are off.
- Nothing is ever submitted to a platform. X only takes a web form that a person fills in; in test
  mode (`DRY_RUN=1`, the default) messages start with "Test:" and escalations aren't filed.

Every session is logged as a `caseworker` event on the case: why it ran, the Droid session id, each
tool call, and the agent's closing summary. Tool effects are logged too (`resolved`, `escalated`).

### Running it

1. Install the droid CLI (`curl -fsSL https://app.factory.ai/cli | sh`) and the API's requirements,
   which include `droid-sdk[mcp]`.
2. Set `FACTORY_API_KEY` (or log in with `droid`). `CASEWORKER=0` turns the agent off;
   `CASEWORKER_MODEL` picks the model (default `auto`, Factory's router).
3. For a demo, confirm a case. With `NOTICE_DEADLINE_SECONDS=5` Factory waits in the same session.
   **Skip the 48-hour wait (test)** on the dashboard (or `POST /timewarp/{case_id}`) cuts the wait
   short, checks X, and prepares the FTC packet.

On Modal, the image installs the CLI, the `factory` secret holds `FACTORY_API_KEY`, and
`deadline_sweep` hits `POST /recheck` hourly to wake sessions if the container scaled down.
`api/test_caseworker.py` and the time-warp block in `api/test_scan.py` check this with a scripted
stand-in for Droid and X, so they need no key.
