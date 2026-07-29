#!/usr/bin/env python3
"""Generate 1000 English users with posts across the three AI topics.

Each user authors one post per topic (3 posts), so every user appears in
different topics. English only for now — user/post records carry `language`
so other locales can be added later without a schema change.

Reuses the English template pool from `generate_toy_posts.py` and applies
light deterministic remixing so texts vary while staying on-topic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_toy_posts import ENGLISH_POSTS, TOPICS

DATA_DIR = ROOT / "data"

N_USERS = 1000
LANGUAGE = "en"
LANGUAGE_NAME = "English"
SEED = 42

FIRST_NAMES = [
    "Alex", "Jordan", "Sam", "Taylor", "Casey", "Riley", "Morgan", "Avery",
    "Quinn", "Reese", "Cameron", "Drew", "Jamie", "Kendall", "Parker", "Finley",
    "Harper", "Rowan", "Skyler", "Dakota", "Emerson", "Hayden", "Logan", "Noah",
    "Olivia", "Emma", "Liam", "Mia", "Ethan", "Sophia", "Lucas", "Isabella",
    "Mason", "Amelia", "James", "Charlotte", "Benjamin", "Harper", "Henry",
    "Evelyn", "Sebastian", "Abigail", "Jack", "Emily", "Owen", "Elizabeth",
    "Daniel", "Sofia", "Matthew", "Aria", "Joseph", "Scarlett", "David",
    "Grace", "Samuel", "Chloe", "John", "Victoria", "Wyatt", "Riley",
    "Luke", "Nora", "Jayden", "Lily", "Dylan", "Eleanor", "Grayson", "Hannah",
    "Levi", "Lillian", "Isaac", "Addison", "Gabriel", "Aubrey", "Julian",
    "Ellie", "Mateo", "Stella", "Anthony", "Natalie", "Jaxon", "Zoe",
    "Lincoln", "Leah", "Joshua", "Hazel", "Christopher", "Violet", "Andrew",
    "Aurora", "Theodore", "Savannah", "Caleb", "Audrey", "Ryan", "Brooklyn",
    "Nathan", "Bella", "Adrian", "Claire", "Nolan", "Skylar", "Aaron", "Lucy",
]

LAST_NAMES = [
    "Rivera", "Nguyen", "Patel", "Brooks", "Chen", "Garcia", "Murphy", "Kim",
    "Singh", "Walsh", "Torres", "Hughes", "Reed", "Bailey", "Cooper", "Richardson",
    "Cox", "Howard", "Ward", "Torres", "Peterson", "Gray", "Ramirez", "James",
    "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes",
    "Ross", "Henderson", "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson",
    "Hughes", "Flores", "Washington", "Butler", "Simmons", "Foster", "Gonzales",
    "Bryant", "Alexander", "Russell", "Griffin", "Diaz", "Hayes", "Myers", "Ford",
    "Hamilton", "Graham", "Sullivan", "Wallace", "Woods", "Cole", "West", "Jordan",
    "Owens", "Reynolds", "Fisher", "Ellis", "Harrison", "Gibson", "Mcdonald",
    "Cruz", "Marshall", "Ortiz", "Gomez", "Murray", "Freeman", "Wells", "Webb",
    "Simpson", "Stevens", "Tucker", "Porter", "Hunter", "Hicks", "Crawford",
    "Henry", "Boyd", "Mason", "Morales", "Kennedy", "Warren", "Dixon", "Ramos",
    "Reyes", "Burns", "Gordon", "Shaw", "Holmes", "Rice", "Robertson", "Hunt",
]

OPENERS = [
    "",
    "Unpopular opinion: ",
    "Quick thought — ",
    "Not gonna lie: ",
    "Real talk: ",
    "Honestly, ",
    "Today's take: ",
    "Just saying — ",
    "Worth repeating: ",
    "Hot take from me: ",
]

CLOSERS = [
    "",
    " Thoughts?",
    " Anyone else seeing this?",
    " Change my mind.",
    " That's the whole story.",
    " Still processing it.",
    " Bookmarking this for later.",
    " Curious what others think.",
    " End of rant.",
    " More soon.",
]


def _rng_for(*parts: str | int) -> random.Random:
    material = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def make_username(display_name: str, user_num: int, rng: random.Random) -> str:
    base = display_name.lower().replace(" ", "_")
    suffix = rng.choice(["", f"_{user_num}", f"{user_num}", f"_{rng.randint(10, 99)}"])
    return f"{base}{suffix}"


def remix_text(template: str, rng: random.Random) -> str:
    """Light remix so users don't all share identical strings."""
    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    text = template
    # Occasionally trim a trailing sentence for variety.
    if rng.random() < 0.25 and ". " in text:
        parts = text.split(". ")
        if len(parts) > 1:
            text = ". ".join(parts[:-1]) + "."
    # Avoid double punctuation / awkward spacing when opener is empty.
    if opener and text and text[0].islower():
        text = text[0].upper() + text[1:]
    composed = f"{opener}{text}{closer}".strip()
    return composed


def generate_users(n: int = N_USERS) -> list[dict]:
    users: list[dict] = []
    used_usernames: set[str] = set()
    for i in range(1, n + 1):
        rng = _rng_for(SEED, "user", i)
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        display_name = f"{first} {last}"
        username = make_username(display_name, i, rng)
        while username in used_usernames:
            username = f"{username}_{rng.randint(100, 999)}"
        used_usernames.add(username)
        users.append(
            {
                "user_id": f"user_{i:04d}",
                "username": username,
                "display_name": display_name,
                "language": LANGUAGE,
                "language_name": LANGUAGE_NAME,
            }
        )
    return users


def generate_posts(users: list[dict]) -> list[dict]:
    """One post per topic per user → every user has posts in different topics."""
    topic_ids = list(TOPICS.keys())
    posts: list[dict] = []
    for user in users:
        for topic_id in topic_ids:
            templates = ENGLISH_POSTS[topic_id]
            rng = _rng_for(SEED, user["user_id"], topic_id)
            template = rng.choice(templates)
            text = remix_text(template, rng)
            # Stable per-user-per-topic id (one post each for now).
            post_id = f"{user['user_id']}_{topic_id}_001"
            posts.append(
                {
                    "post_id": post_id,
                    "user_id": user["user_id"],
                    "topic": topic_id,
                    "topic_label": TOPICS[topic_id]["label"],
                    "language": LANGUAGE,
                    "language_name": LANGUAGE_NAME,
                    "text": text,
                }
            )
    return posts


def write_users(users: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = DATA_DIR / "users.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for user in users:
            f.write(json.dumps(user, ensure_ascii=False) + "\n")

    nested = {
        "description": (
            "Synthetic English-speaking users for toy embedding / clustering "
            "experiments. Each user authors posts across multiple topics."
        ),
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "n_users": len(users),
        "topics": TOPICS,
        "users": users,
    }
    json_path = DATA_DIR / "users.json"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "users.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["user_id", "username", "display_name", "language", "language_name"],
        )
        writer.writeheader()
        writer.writerows(users)

    print(f"Wrote {jsonl_path} ({len(users)} users)")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def write_user_posts(posts: list[dict], n_users: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = DATA_DIR / "user_posts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in posts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_user: dict[str, list[dict]] = {}
    for row in posts:
        by_user.setdefault(row["user_id"], []).append(
            {
                "post_id": row["post_id"],
                "topic": row["topic"],
                "topic_label": row["topic_label"],
                "language": row["language"],
                "text": row["text"],
            }
        )

    nested = {
        "description": (
            "English-only posts authored by synthetic users. Each user has one "
            "post in each topic so authorship spans topics."
        ),
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "n_users": n_users,
        "n_posts": len(posts),
        "n_topics": len(TOPICS),
        "posts_per_user": len(TOPICS),
        "topics": TOPICS,
        "posts_by_user": by_user,
    }
    json_path = DATA_DIR / "user_posts.json"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "user_posts.csv"
    fieldnames = [
        "post_id",
        "user_id",
        "topic",
        "topic_label",
        "language",
        "language_name",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(posts)

    print(f"Wrote {jsonl_path} ({len(posts)} posts)")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def update_data_readme(n_users: int, n_posts: int) -> None:
    """Append / refresh the English-users section without wiping multilingual docs."""
    readme_path = DATA_DIR / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    marker = "## English users (synthetic)"
    section = (
        f"{marker}\n\n"
        "1000 English-only synthetic users, each with one post per topic "
        f"({len(TOPICS)} topics → {n_posts} posts). Language fields are present "
        "so non-English users can be added later.\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "python scripts/generate_english_users.py\n"
        "```\n\n"
        "### Scale\n\n"
        f"- **{n_users}** users (`language=en`)\n"
        f"- **{n_posts}** posts (1 per topic per user)\n"
        "- Topics: same three as the multilingual toy set\n\n"
        "### Files\n\n"
        "| File | Format |\n| --- | --- |\n"
        "| `users.json` | Nested user list + metadata |\n"
        "| `users.jsonl` | One user per line |\n"
        "| `users.csv` | Flat users |\n"
        "| `user_posts.json` | Posts nested by `user_id` |\n"
        "| `user_posts.jsonl` | Flat posts with `user_id` |\n"
        "| `user_posts.csv` | Flat CSV |\n\n"
        "### User schema\n\n"
        "`user_id`, `username`, `display_name`, `language`, `language_name`\n\n"
        "### Post schema\n\n"
        "`post_id`, `user_id`, `topic`, `topic_label`, `language`, `language_name`, `text`\n"
    )

    if marker in existing:
        head, _sep, rest = existing.partition(marker)
        # Drop old section through EOF (this section is appended last).
        existing = head.rstrip() + "\n\n" + section
    else:
        existing = existing.rstrip() + "\n\n" + section

    readme_path.write_text(existing, encoding="utf-8")
    print(f"Updated {readme_path}")


def main() -> None:
    users = generate_users(N_USERS)
    posts = generate_posts(users)

    # Sanity: every user covers all topics.
    topic_ids = set(TOPICS)
    by_user: dict[str, set[str]] = {}
    for p in posts:
        by_user.setdefault(p["user_id"], set()).add(p["topic"])
    bad = [uid for uid, topics in by_user.items() if topics != topic_ids]
    if bad:
        raise SystemExit(f"{len(bad)} users missing topics, e.g. {bad[:3]}")

    write_users(users)
    write_user_posts(posts, n_users=len(users))
    update_data_readme(n_users=len(users), n_posts=len(posts))
    print(f"Done. {len(users)} English users, {len(posts)} posts.")


if __name__ == "__main__":
    main()
