"""Takedown channels (data/channels.json), the TAKE IT DOWN notice (templates/notice.md), and email.

C owns both files; templates/NOTES.md maps every placeholder and lists the legal rules that
TakedownNotice and main.notice() enforce. Six Eyes files as the user's authorized representative:
a named team member signs (SIGNER_NAME) and replies go to the team's reporting inbox (CONTACT_EMAIL).

Email settings come from the environment: SMTP_HOST, SMTP_PORT (587 STARTTLS, 465 SSL), SMTP_USER,
SMTP_PASSWORD and NOTICE_FROM (defaults to SMTP_USER).
"""
import json
import os
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator

CHANNELS_PATH = Path(__file__).with_name("data") / "channels.json"
TEMPLATES = Path(__file__).with_name("templates")
HOST_ALIASES = {"twitter.com": "x", "mobile.twitter.com": "x", "x.com": "x", "mobile.x.com": "x"}
RENDERED = re.compile(r"<rendered api/templates/(\w+\.md)>")  # channels.json's stand-in for a whole template
# depiction_kind values, from NOTES.md
FORGERY = 'a digitally altered or AI-generated image of them (a "digital forgery" as defined in the Act)'
INTIMATE = "an intimate image of them"


class FormField(BaseModel):
    question: str
    required: bool = False
    type: str = "text"
    value: str  # a literal answer, or text with notice.md placeholders
    preset: bool = False  # the channel's url already picks this answer


class Checkbox(BaseModel):
    label: str
    check: bool
    why: str = ""


class Channel(BaseModel):
    id: str
    name: str
    method: str  # web_form | in_app | hash_submission
    url: str | None = None
    in_app_path: str | None = None
    verified: bool = False
    fields: list[FormField] = []
    checkboxes: list[Checkbox] = []


class TakedownNotice(BaseModel):
    """Everything notice.md needs, so a notice missing a required element of Act 3(a)(1)(B) can't be built."""
    case_id: int
    platform_name: str = Field(min_length=1)
    user_name: str = Field(min_length=1)
    authorization_date: datetime              # users.signed_at: when the user authorized us to file
    reported_username: str = Field(pattern=r"^\w{1,15}$")  # the posting account, without "@"
    content_urls: list[str] = Field(min_length=1)  # every post by that account in the clone cluster
    depiction_kind: Literal[FORGERY, INTIMATE]
    signer_name: str = Field(min_length=1)     # signature: a real person who stands behind the statement
    signer_relationship: str = Field(min_length=1)
    contact_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    non_consent_confirmed: Literal[True]      # the user pressed "This is me" and "Not consensual"

    @field_validator("content_urls")
    @classmethod
    def _urls(cls, urls: list[str]) -> list[str]:
        if bad := [u for u in urls if urlsplit(u).scheme not in ("http", "https") or not urlsplit(u).hostname]:
            raise ValueError(f"not a web address: {', '.join(bad)}")
        return list(dict.fromkeys(urls))

    def placeholders(self, now: datetime) -> dict[str, str]:
        return {
            "notice_date": _date(now), "platform_name": self.platform_name, "case_id": str(self.case_id),
            "user_name": self.user_name, "authorization_date": _date(self.authorization_date),
            "reported_username": self.reported_username,
            "content_urls": "\n".join(f"- {u}" for u in self.content_urls),
            "depiction_kind": self.depiction_kind, "signer_name": self.signer_name,
            "signer_relationship": self.signer_relationship, "contact_email": self.contact_email,
        }


@lru_cache(maxsize=1)
def channels() -> dict[str, Channel]:
    return {p["id"]: Channel(**p) for p in json.loads(CHANNELS_PATH.read_text())["platforms"]}


@lru_cache(maxsize=1)
def escalation() -> dict[str, str]:
    """Where to go when a platform misses the deadline: "ftc" (adults), "minors" (NCMEC)."""
    return json.loads(CHANNELS_PATH.read_text())["escalation"]


def channel_for(url: str) -> Channel:
    """The takedown channel for the platform a post URL is on; KeyError if there isn't one."""
    host = (urlsplit(url).hostname or "").removeprefix("www.")
    return channels()[HOST_ALIASES.get(host, host)]


def render(notice: TakedownNotice, now: datetime, template: str = "notice.md") -> str:
    return (TEMPLATES / template).read_text().strip().format(**notice.placeholders(now))


def form_answers(channel: Channel, notice: TakedownNotice, now: datetime) -> list[dict]:
    """The channel's form, question by question, with the answer for this notice filled in."""
    values = notice.placeholders(now)
    return [{"question": f.question, "required": f.required, "type": f.type, "preset": f.preset,
             "value": render(notice, now, m[1]) if (m := RENDERED.fullmatch(f.value)) else f.value.format(**values)}
            for f in channel.fields]


def instructions(channel: Channel, packet: dict) -> str:
    """Plain-text steps for filing one packet through the channel's form, for the test email."""
    lines = [f"Report @{packet['reported_username']} on {channel.name}", f"Form: {channel.url}", ""]
    for i, f in enumerate(packet["form"], 1):
        answer = f["value"]
        if f["preset"]:
            answer += " (the form link already picks this)"
        lines += [f"{i}. {f['question']}{'' if f['required'] else ' (optional)'}", f"   {answer}"]
    lines.append("")
    for box in packet["checkboxes"]:
        lines.append(f"[{'x' if box['check'] else ' '}] {box['label']}")
    return "\n".join(lines + ["", "Notice:", "", packet["notice"]])


def send_email(to: str, subject: str, body: str, reply_to: str | None = None) -> None:
    host, user = os.environ.get("SMTP_HOST"), os.environ.get("SMTP_USER")
    sender = os.environ.get("NOTICE_FROM") or user
    if not (host and sender):
        raise RuntimeError("Email isn't set up: set SMTP_HOST, SMTP_USER and SMTP_PASSWORD on the API.")
    port = int(os.environ.get("SMTP_PORT", "587"))
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.set_content(body)
    with (smtplib.SMTP_SSL if port == 465 else smtplib.SMTP)(host, port, timeout=20) as smtp:
        if port != 465:
            smtp.starttls()
        if user:
            smtp.login(user, os.environ.get("SMTP_PASSWORD", ""))
        smtp.send_message(msg)


def _date(d: datetime) -> str:
    return f"{d:%B} {d.day}, {d:%Y}"
