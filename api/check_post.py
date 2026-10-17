"""Why was (or wasn't) a post found? Run from the repo root with X_BEARER_TOKEN set:
    .venv/bin/python api/check_post.py <post url or id> <user_id>
Reads one post from X (one billed read) and face-matches its images against the user's enrolled face.
"""
import json
import re
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone

import analyze
import signals
import x
from main import FACE_MATCH_MIN, SWEEP_LIMIT, db

X_EPOCH_MS = 1288834974657


def posted_at(post_id: str) -> datetime:
    return datetime.fromtimestamp(((int(post_id) >> 22) + X_EPOCH_MS) / 1000, timezone.utc)


def mentions(text: str, phrase: str) -> bool:
    words = r"[\W_]+".join(map(re.escape, phrase.split()))
    return re.search(rf"(?<!\w){words}(?!\w)", text, re.I) is not None


def main(ref: str, user_id: int) -> None:
    post_id = m[1] if (m := re.search(r"/status/(\d+)", ref)) else ref.strip()
    with closing(db()) as con:
        user = con.execute("SELECT name, handles, embedding FROM users WHERE id = ?", (user_id,)).fetchone()
        sweep = con.execute("SELECT since_id FROM sweeps WHERE user_id = ?", (user_id,)).fetchone()
        saved = con.execute("SELECT COUNT(*) FROM cases WHERE user_id = ? AND post_id = ?", (user_id, post_id)).fetchone()[0]
    if not user:
        sys.exit(f"No enrolled user with id {user_id}.")
    name, handles, embedding = user[0], json.loads(user[1]), json.loads(user[2])

    post = x.lookup(post_id)
    if not post:
        sys.exit("X didn't return this post with an image or video: it's deleted, private, or text-only.")
    when = posted_at(post_id)
    print(f"Post: {post.url}\nText: {post.text!r}\nPosted: {when.astimezone():%b %d %H:%M %Z}"
          f"{'  (older than 7 days: outside X recent search)' if datetime.now(timezone.utc) - when > timedelta(days=7) else ''}")
    print(f"Already a case for this user: {'yes' if saved else 'no'}")

    by_name = [t for t in [name, *(f"@{h}" for h in handles)] if mentions(post.text, t.lstrip("@"))]
    by_term = [t for t in signals.search_terms() if mentions(post.text, t)]
    print(f"\nName search: {'matches ' + ', '.join(by_name) if by_name else 'no match (no name or @handle in the text)'}")
    print(f"Keyword search: {'matches ' + ', '.join(by_term) if by_term else 'no match (no search term in the text)'}")
    if by_term:
        if not sweep:
            print("  No keyword search has run for this user yet; the next scan with sweep=true will check it.")
        elif int(post_id) > int(sweep[0]):
            print("  Newer than the last keyword search; the next scan with sweep=true will check it.")
        else:
            print(f"  The last keyword search started from a post at {posted_at(sweep[0]).astimezone():%b %d %H:%M %Z} "
                  f"and read back at most {SWEEP_LIMIT} posts.\n  If more than {SWEEP_LIMIT} posts using those terms "
                  "were made between then and this post, the search never reached it.")

    print(f"\nFace match (a keyword-search hit needs {FACE_MATCH_MIN} or more):")
    for media in post.media:
        try:
            sim = analyze.analyze(media.url, embedding).face_sim
            print(f"  {media.type} {media.url}: {sim:.3f} {'match' if sim >= FACE_MATCH_MIN else 'no match'}")
        except Exception as e:
            print(f"  {media.type} {media.url}: couldn't check ({e})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]))
