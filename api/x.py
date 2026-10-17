"""X API v2 client: find posts that may show the user, and check whether a post is still up.

X can't search by image, so we find candidate images two ways and let analyze() face-match them:
- search() finds posts that name or mention one user.
- sweep() finds posts that use deepfake terms, whoever they show, for matching against every user.
Needs X_BEARER_TOKEN.
"""
import os
import re
import sys

import httpx
from pydantic import BaseModel

API = "https://api.x.com/2"
MAX_QUERY = 512  # pay-per-use limit; Enterprise allows 4,096
HANDLE = re.compile(r"[A-Za-z0-9_]{1,15}")
MEDIA_ONLY = "(has:images OR has:videos) -is:retweet"
MAX_VIDEO_BITRATE = 2_500_000  # X's 720p MP4 is ~2.2 Mbps; 1080p is ~10 Mbps and slow to download
SEARCH_FIELDS = {
    "expansions": "attachments.media_keys,author_id",
    "tweet.fields": "created_at,public_metrics,possibly_sensitive,entities,attachments,author_id",
    "media.fields": "type,url,variants",
    "user.fields": "created_at,public_metrics,description",
}


class Author(BaseModel):
    id: str
    username: str
    name: str
    created_at: str | None = None
    description: str = ""
    public_metrics: dict[str, int] = {}  # followers_count, following_count, tweet_count, ...


class Media(BaseModel):
    key: str
    type: str  # photo | video | animated_gif
    url: str   # photo: the image; video and GIF: an MP4 of at most ~720p, enough for face matching


class Post(BaseModel):
    id: str
    url: str
    text: str
    created_at: str | None = None
    possibly_sensitive: bool = False
    public_metrics: dict[str, int] = {}  # like_count, retweet_count, reply_count, ...
    entities: dict = {}                  # hashtags, urls, mentions
    author: Author | None = None
    media: list[Media]


def query(name: str, handles: list[str]) -> str:
    terms = [f'"{n}"'] if (n := name.replace('"', "").strip()) else []
    for h in dict.fromkeys(h.lstrip("@").lower() for h in handles):
        if HANDLE.fullmatch(h):
            terms += [f"@{h}", f"to:{h}"]
    if not terms:
        raise ValueError("Need a name or at least one valid X handle to search.")
    return _media_query(terms)


def keyword_query(terms: list[str]) -> str:
    phrases = dict.fromkeys(t.replace('"', "").strip().lower() for t in terms)
    if not (quoted := [f'"{p}"' for p in phrases if p]):
        raise ValueError("Need at least one search term.")
    return _media_query(quoted)


def search(name: str, handles: list[str], limit: int = 100, since_id: str | None = None) -> list[Post]:
    """Recent (last 7 days) posts with images or videos that name or mention the user.

    Pass the newest post id from the last scan as since_id so X doesn't bill for posts already seen.
    """
    return _search(query(name, handles), limit, since_id)


def sweep(terms: list[str], limit: int = 500, since_id: str | None = None) -> list[Post]:
    """Recent posts with images or videos that use any of the terms (signals.search_terms()).

    Not tied to one user: run it once per scan cycle and face-match every result against every
    enrolled user. X matches each term as a phrase on word boundaries, so "deepfake" also matches
    #deepfake and "AI generated" also matches AI-generated.
    """
    return _search(keyword_query(terms), limit, since_id)


def _media_query(terms: list[str]) -> str:
    q = f"({' OR '.join(terms)}) {MEDIA_ONLY}"
    if len(q) > MAX_QUERY:
        raise ValueError(f"X search query is {len(q)} characters; the limit is {MAX_QUERY}.")
    return q


def _search(q: str, limit: int, since_id: str | None) -> list[Post]:
    params = {"query": q, **SEARCH_FIELDS}
    if since_id:
        params["since_id"] = since_id
    posts: list[Post] = []
    while len(posts) < limit:
        params["max_results"] = min(100, max(10, limit - len(posts)))
        page = _get("/tweets/search/recent", params)
        posts += _posts(page)
        if not (token := page.get("meta", {}).get("next_token")):
            break
        params["next_token"] = token
    return posts[:limit]


def is_live(post_id: str) -> bool:
    """False once the post is deleted, or hidden because its author was suspended or went private."""
    return "data" in _get(f"/tweets/{post_id}", {})


def _get(path: str, params: dict) -> dict:
    token = os.environ.get("X_BEARER_TOKEN")
    if not token:
        raise RuntimeError("X_BEARER_TOKEN is not set.")
    r = httpx.get(API + path, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=20)
    r.raise_for_status()
    return r.json()


def _posts(page: dict) -> list[Post]:
    includes = page.get("includes", {})
    users = {u["id"]: Author(**u) for u in includes.get("users", [])}
    media = {m["media_key"]: m for m in includes.get("media", [])}
    posts = []
    for t in page.get("data", []):
        items = [
            Media(key=k, type=m["type"], url=url)
            for k in t.get("attachments", {}).get("media_keys", [])
            if (m := media.get(k)) and (url := _media_url(m))
        ]
        if not items:
            continue
        author = users.get(t.get("author_id"))
        posts.append(Post(
            id=t["id"],
            url=f"https://x.com/{author.username if author else 'i'}/status/{t['id']}",
            text=t["text"],
            created_at=t.get("created_at"),
            possibly_sensitive=t.get("possibly_sensitive", False),
            public_metrics=t.get("public_metrics", {}),
            entities=t.get("entities", {}),
            author=author,
            media=items,
        ))
    return posts


def _media_url(m: dict) -> str | None:
    if m["type"] == "photo":
        return m.get("url")
    mp4s = sorted((v for v in m.get("variants", []) if v.get("content_type") == "video/mp4"),
                  key=lambda v: v.get("bit_rate", 0))
    if not mp4s:
        return None
    small = [v for v in mp4s if v.get("bit_rate", 0) <= MAX_VIDEO_BITRATE]
    return (small[-1] if small else mp4s[0])["url"]


if __name__ == "__main__":
    # Smoke test, reads at most 10 posts:  python api/x.py "Full Name" handle [handle ...]
    print("Query:", query(sys.argv[1], sys.argv[2:]))
    found = search(sys.argv[1], sys.argv[2:], limit=10)
    for p in found:
        print(f"\n{p.url}  (by @{p.author.username if p.author else '?'})\n  {p.text[:100]!r}")
        for m in p.media:
            print(f"  {m.type}: {m.url}")
    print(f"\n{len(found)} post(s) with images or videos found.")
