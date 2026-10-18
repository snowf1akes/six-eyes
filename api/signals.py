"""Context signals for a found post: what it says, who posted it, how far it spread, and which
other posts share the same image (clone clusters).

C owns data/keywords.json: {"search": [...], "keywords": [...], "hashtags": [...], "domains": [...]}.
"""
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel

from x import Author, Post

KEYWORDS_PATH = Path(__file__).with_name("data") / "keywords.json"
CLONE_MAX_DISTANCE = 31  # PDQ Hamming distance (of 256 bits) at or under which two images match


class Signals(BaseModel):
    keywords: list[str]               # entries from keywords.json found in the post text
    hashtags: list[str]               # flagged hashtags on the post, lowercase, without '#'
    domains: list[str]                # flagged domains the post links to
    author_age_days: float | None     # None when X didn't return the author
    follower_ratio: float | None      # followers / following; burner accounts sit near 0
    engagement: dict[str, int]        # like, retweet, reply, quote, bookmark, impression counts


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads(KEYWORDS_PATH.read_text())


def search_terms() -> list[str]:
    """Terms for x.sweep(), which finds posts by topic instead of by the user's name."""
    return [t for t in _data().get("search", []) if t.strip()]


@lru_cache(maxsize=1)
def flag_lists() -> tuple[re.Pattern | None, frozenset[str], tuple[str, ...]]:
    """Search terms are flagged too, in text and as hashtags ("AI generated" also flags #AIgenerated)."""
    data = _data()
    terms = search_terms()
    words = sorted({k.strip() for k in data.get("keywords", []) + terms if k.strip()}, key=len, reverse=True)
    pattern = re.compile(rf"(?<!\w)(?:{'|'.join(map(re.escape, words))})(?!\w)", re.I) if words else None
    tags = frozenset(t.strip().lstrip("#").lower() for t in data.get("hashtags", []) if t.strip()) | {
        re.sub(r"[\s-]+", "", t).lower() for t in terms}
    domains = tuple({d.strip().lower().removeprefix("www.") for d in data.get("domains", []) if d.strip()})
    return pattern, tags, domains


def signals(post: Post, now: datetime | None = None) -> Signals:
    pattern, tags, domains = flag_lists()
    keywords = sorted({m.lower() for m in pattern.findall(post.text)}) if pattern else []
    hashtags = sorted({h["tag"].lower() for h in post.entities.get("hashtags", [])} & tags)
    hosts = {_host(u.get("unwound_url") or u.get("expanded_url") or u.get("url", ""))
             for u in post.entities.get("urls", [])}
    linked = sorted({d for d in domains for h in hosts if h == d or h.endswith("." + d)})
    age, ratio = _author(post.author, now or datetime.now(timezone.utc))
    engagement = {k.removesuffix("_count"): v for k, v in post.public_metrics.items()}
    return Signals(keywords=keywords, hashtags=hashtags, domains=linked,
                   author_age_days=age, follower_ratio=ratio, engagement=engagement)


def clusters(pdqs: list[str | None]) -> list[int | None]:
    """For each hash, the index of the first hash in its clone cluster (None if it has no hash).

    Matches chain: if A matches B and B matches C, all three share a cluster. Pass the user's
    existing cases oldest first, then the new ones, so earlier cluster ids stay put.
    """
    bits = [int(p, 16) if p else None for p in pdqs]
    root = list(range(len(bits)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for i, a in enumerate(bits):
        for j in range(i):
            if a is not None and bits[j] is not None and (a ^ bits[j]).bit_count() <= CLONE_MAX_DISTANCE:
                ri, rj = find(i), find(j)
                root[max(ri, rj)] = min(ri, rj)
    return [find(i) if b is not None else None for i, b in enumerate(bits)]


def _author(author: Author | None, now: datetime) -> tuple[float | None, float | None]:
    if not author:
        return None, None
    age = None
    if author.created_at:
        created = datetime.fromisoformat(author.created_at.replace("Z", "+00:00"))
        age = round((now - created).total_seconds() / 86400, 1)
    m = author.public_metrics
    ratio = None
    if "followers_count" in m:
        ratio = round(m["followers_count"] / max(m.get("following_count", 0), 1), 3)
    return age, ratio


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").removeprefix("www.")
