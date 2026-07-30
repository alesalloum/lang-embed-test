#!/usr/bin/env python3
"""Generate English users with known topic-interest profiles and their posts.

Purpose
-------
Build a labeled toy set for recovering user interest types from posts, e.g.:
users who mostly talk about topic 1 vs topic 2 vs users who talk evenly, etc.

Each user is assigned a predefined **profile** with a ground-truth
`topic_distribution` over the three AI topics. Posts are sampled i.i.d. from
that distribution (100 posts / user by default). Both the generative
distribution and the realized empirical counts are stored on the user record.

English only for now; `language` fields remain so other locales can be added
later without a schema change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_toy_posts import ENGLISH_POSTS, TOPICS

DATA_DIR = ROOT / "data"

N_USERS = 1000
POSTS_PER_USER = 100
LANGUAGE = "en"
LANGUAGE_NAME = "English"
SEED = 42

TOPIC_IDS = list(TOPICS.keys())  # coding, copyright, surveillance

# ---------------------------------------------------------------------------
# Predefined user profiles = ground-truth interest distributions
# ---------------------------------------------------------------------------
# Probabilities are ordered to match TOPIC_IDS. They must sum to 1.
USER_PROFILES: dict[str, dict] = {
    "coding_heavy": {
        "label": "Mostly AI coding innovations",
        "description": "Dominantly posts about AI coding tools; rare other topics.",
        "topic_distribution": {
            "ai_coding_innovations": 0.80,
            "ai_copyright_theft": 0.10,
            "ai_mass_surveillance": 0.10,
        },
    },
    "copyright_heavy": {
        "label": "Mostly AI copyright / theft concerns",
        "description": "Dominantly posts about generative-AI copyright theft.",
        "topic_distribution": {
            "ai_coding_innovations": 0.10,
            "ai_copyright_theft": 0.80,
            "ai_mass_surveillance": 0.10,
        },
    },
    "surveillance_heavy": {
        "label": "Mostly AI mass surveillance",
        "description": "Dominantly posts about AI-enabled surveillance.",
        "topic_distribution": {
            "ai_coding_innovations": 0.10,
            "ai_copyright_theft": 0.10,
            "ai_mass_surveillance": 0.80,
        },
    },
    "balanced": {
        "label": "Even across all topics",
        "description": "Posts roughly evenly about all three topics.",
        "topic_distribution": {
            "ai_coding_innovations": 1 / 3,
            "ai_copyright_theft": 1 / 3,
            "ai_mass_surveillance": 1 / 3,
        },
    },
    "coding_copyright": {
        "label": "Split: coding + copyright",
        "description": "Cares about coding and copyright; rarely surveillance.",
        "topic_distribution": {
            "ai_coding_innovations": 0.45,
            "ai_copyright_theft": 0.45,
            "ai_mass_surveillance": 0.10,
        },
    },
    "coding_surveillance": {
        "label": "Split: coding + surveillance",
        "description": "Cares about coding and surveillance; rarely copyright.",
        "topic_distribution": {
            "ai_coding_innovations": 0.45,
            "ai_copyright_theft": 0.10,
            "ai_mass_surveillance": 0.45,
        },
    },
    "copyright_surveillance": {
        "label": "Split: copyright + surveillance",
        "description": "Cares about copyright and surveillance; rarely coding.",
        "topic_distribution": {
            "ai_coding_innovations": 0.10,
            "ai_copyright_theft": 0.45,
            "ai_mass_surveillance": 0.45,
        },
    },
}

PROFILE_IDS = list(USER_PROFILES.keys())

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
    "Cox", "Howard", "Ward", "Peterson", "Gray", "Ramirez", "James", "Watson",
    "Kelly", "Sanders", "Price", "Bennett", "Wood", "Barnes", "Ross", "Henderson",
    "Coleman", "Jenkins", "Perry", "Powell", "Long", "Patterson", "Flores",
    "Washington", "Butler", "Simmons", "Foster", "Gonzales", "Bryant", "Alexander",
    "Russell", "Griffin", "Diaz", "Hayes", "Myers", "Ford", "Hamilton", "Graham",
    "Sullivan", "Wallace", "Woods", "Cole", "West", "Jordan", "Owens", "Reynolds",
    "Fisher", "Ellis", "Harrison", "Gibson", "Mcdonald", "Cruz", "Marshall",
    "Ortiz", "Gomez", "Murray", "Freeman", "Wells", "Webb", "Simpson", "Stevens",
    "Tucker", "Porter", "Hunter", "Hicks", "Crawford", "Henry", "Boyd", "Mason",
    "Morales", "Kennedy", "Warren", "Dixon", "Ramos", "Reyes", "Burns", "Gordon",
    "Shaw", "Holmes", "Rice", "Robertson", "Hunt",
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
    "Thread: ",
    "Hearing this a lot lately: ",
    "For what it's worth: ",
    "Slightly spicy take: ",
    "Logging this: ",
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
    " Am I wrong?",
    " Saying it louder for the people in the back.",
    " Anyway, that's my two cents.",
    " Circling back later.",
]

ASIDES = [
    "",
    " (and yes, I checked)",
    " — wild times",
    " (serious)",
    " tbh",
    " imo",
    " — not hyperbolic",
    " (again)",
]


def _rng_for(*parts: str | int) -> random.Random:
    material = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def _round_dist(dist: dict[str, float], ndigits: int = 6) -> dict[str, float]:
    """Round probs and nudge the last topic so the vector still sums to 1."""
    keys = list(dist.keys())
    rounded = {k: round(float(dist[k]), ndigits) for k in keys[:-1]}
    rounded[keys[-1]] = round(1.0 - sum(rounded.values()), ndigits)
    return rounded


def validate_profiles() -> None:
    for pid, profile in USER_PROFILES.items():
        dist = profile["topic_distribution"]
        if set(dist) != set(TOPIC_IDS):
            raise SystemExit(f"Profile {pid}: topic keys must match TOPIC_IDS")
        total = sum(dist.values())
        if abs(total - 1.0) > 1e-9:
            raise SystemExit(f"Profile {pid}: distribution sums to {total}, not 1")
        if any(v < 0 for v in dist.values()):
            raise SystemExit(f"Profile {pid}: negative probability")


def assign_profile_id(user_index: int) -> str:
    """Round-robin so profiles are as balanced as possible across N_USERS."""
    return PROFILE_IDS[(user_index - 1) % len(PROFILE_IDS)]


def make_username(display_name: str, user_num: int, rng: random.Random) -> str:
    base = display_name.lower().replace(" ", "_")
    suffix = rng.choice(["", f"_{user_num}", f"{user_num}", f"_{rng.randint(10, 99)}"])
    return f"{base}{suffix}"


def remix_text(template: str, rng: random.Random) -> str:
    """Light remix so repeated templates still differ across 100 posts/user."""
    opener = rng.choice(OPENERS)
    closer = rng.choice(CLOSERS)
    aside = rng.choice(ASIDES)
    text = template
    if rng.random() < 0.20 and ". " in text:
        parts = text.split(". ")
        if len(parts) > 1:
            text = ". ".join(parts[:-1]) + "."
    if opener and text and text[0].islower():
        text = text[0].upper() + text[1:]
    # Insert aside before final punctuation when present.
    if aside and text.endswith((".", "!", "?")):
        text = text[:-1] + aside + text[-1]
    elif aside:
        text = text + aside
    return f"{opener}{text}{closer}".strip()


def sample_topic_sequence(
    topic_distribution: dict[str, float],
    n_posts: int,
    rng: random.Random,
) -> list[str]:
    topics = list(topic_distribution.keys())
    weights = [topic_distribution[t] for t in topics]
    return rng.choices(topics, weights=weights, k=n_posts)


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

        profile_id = assign_profile_id(i)
        profile = USER_PROFILES[profile_id]
        topic_distribution = _round_dist(profile["topic_distribution"])

        users.append(
            {
                "user_id": f"user_{i:04d}",
                "username": username,
                "display_name": display_name,
                "language": LANGUAGE,
                "language_name": LANGUAGE_NAME,
                "profile_id": profile_id,
                "profile_label": profile["label"],
                "topic_distribution": topic_distribution,
                "n_posts": POSTS_PER_USER,
                # Filled after posts are sampled:
                "topic_counts": None,
                "empirical_topic_distribution": None,
            }
        )
    return users


def generate_posts(users: list[dict]) -> list[dict]:
    posts: list[dict] = []
    for user in users:
        dist = user["topic_distribution"]
        rng = _rng_for(SEED, "posts", user["user_id"])
        topic_seq = sample_topic_sequence(dist, POSTS_PER_USER, rng)
        counts = Counter(topic_seq)
        user["topic_counts"] = {t: int(counts.get(t, 0)) for t in TOPIC_IDS}
        user["empirical_topic_distribution"] = _round_dist(
            {t: user["topic_counts"][t] / POSTS_PER_USER for t in TOPIC_IDS}
        )

        per_topic_idx = Counter()
        for topic_id in topic_seq:
            per_topic_idx[topic_id] += 1
            idx = per_topic_idx[topic_id]
            templates = ENGLISH_POSTS[topic_id]
            post_rng = _rng_for(SEED, user["user_id"], topic_id, idx)
            text = remix_text(post_rng.choice(templates), post_rng)
            posts.append(
                {
                    "post_id": f"{user['user_id']}_{topic_id}_{idx:03d}",
                    "user_id": user["user_id"],
                    "profile_id": user["profile_id"],
                    "topic": topic_id,
                    "topic_label": TOPICS[topic_id]["label"],
                    "language": LANGUAGE,
                    "language_name": LANGUAGE_NAME,
                    "text": text,
                }
            )
    return posts


def write_profiles() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Ground-truth user interest profiles. Each profile defines a "
            "topic_distribution used to sample that user's posts. Recovery "
            "experiments should try to rediscover profile_id / distribution "
            "from posts alone."
        ),
        "topics": TOPICS,
        "language": LANGUAGE,
        "posts_per_user": POSTS_PER_USER,
        "profiles": {
            pid: {
                "profile_id": pid,
                "label": p["label"],
                "description": p["description"],
                "topic_distribution": _round_dist(p["topic_distribution"]),
            }
            for pid, p in USER_PROFILES.items()
        },
    }
    path = DATA_DIR / "user_profiles.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path} ({len(USER_PROFILES)} profiles)")


def write_users(users: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = DATA_DIR / "users.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for user in users:
            f.write(json.dumps(user, ensure_ascii=False) + "\n")

    profile_counts = Counter(u["profile_id"] for u in users)
    nested = {
        "description": (
            "Synthetic English users with known ground-truth topic-interest "
            "profiles. Use profile_id / topic_distribution as labels when "
            "testing recovery of user types from posts."
        ),
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "n_users": len(users),
        "posts_per_user": POSTS_PER_USER,
        "topics": TOPICS,
        "profiles": {
            pid: {
                "label": p["label"],
                "topic_distribution": _round_dist(p["topic_distribution"]),
                "n_users": profile_counts[pid],
            }
            for pid, p in USER_PROFILES.items()
        },
        "users": users,
    }
    json_path = DATA_DIR / "users.json"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "users.csv"
    fieldnames = [
        "user_id",
        "username",
        "display_name",
        "language",
        "language_name",
        "profile_id",
        "profile_label",
        "n_posts",
        "p_ai_coding_innovations",
        "p_ai_copyright_theft",
        "p_ai_mass_surveillance",
        "n_ai_coding_innovations",
        "n_ai_copyright_theft",
        "n_ai_mass_surveillance",
        "emp_ai_coding_innovations",
        "emp_ai_copyright_theft",
        "emp_ai_mass_surveillance",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            dist = user["topic_distribution"]
            counts = user["topic_counts"]
            emp = user["empirical_topic_distribution"]
            writer.writerow(
                {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "display_name": user["display_name"],
                    "language": user["language"],
                    "language_name": user["language_name"],
                    "profile_id": user["profile_id"],
                    "profile_label": user["profile_label"],
                    "n_posts": user["n_posts"],
                    "p_ai_coding_innovations": dist["ai_coding_innovations"],
                    "p_ai_copyright_theft": dist["ai_copyright_theft"],
                    "p_ai_mass_surveillance": dist["ai_mass_surveillance"],
                    "n_ai_coding_innovations": counts["ai_coding_innovations"],
                    "n_ai_copyright_theft": counts["ai_copyright_theft"],
                    "n_ai_mass_surveillance": counts["ai_mass_surveillance"],
                    "emp_ai_coding_innovations": emp["ai_coding_innovations"],
                    "emp_ai_copyright_theft": emp["ai_copyright_theft"],
                    "emp_ai_mass_surveillance": emp["ai_mass_surveillance"],
                }
            )

    print(f"Wrote {jsonl_path} ({len(users)} users)")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


def write_user_posts(posts: list[dict], n_users: int) -> None:
    """Write flat post files. Skip a fully nested dump (too large at 100k posts)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    jsonl_path = DATA_DIR / "user_posts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in posts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    topic_counts = Counter(p["topic"] for p in posts)
    profile_counts = Counter(p["profile_id"] for p in posts)
    meta = {
        "description": (
            "English posts sampled from each user's ground-truth "
            "topic_distribution. Flat records live in user_posts.jsonl / .csv."
        ),
        "language": LANGUAGE,
        "language_name": LANGUAGE_NAME,
        "n_users": n_users,
        "n_posts": len(posts),
        "posts_per_user": POSTS_PER_USER,
        "n_topics": len(TOPICS),
        "topics": TOPICS,
        "topic_counts": dict(topic_counts),
        "posts_per_profile": dict(profile_counts),
        "files": {
            "user_posts.jsonl": "One post per line (primary)",
            "user_posts.csv": "Same flat schema as JSONL",
        },
    }
    json_path = DATA_DIR / "user_posts.json"
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = DATA_DIR / "user_posts.csv"
    fieldnames = [
        "post_id",
        "user_id",
        "profile_id",
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
    print(f"Wrote {json_path} (metadata only)")
    print(f"Wrote {csv_path}")


def update_data_readme(n_users: int, n_posts: int) -> None:
    readme_path = DATA_DIR / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    profile_rows = "\n".join(
        "| `{pid}` | {label} | {coding:.2f} / {copyright:.2f} / {surv:.2f} |".format(
            pid=pid,
            label=p["label"],
            coding=p["topic_distribution"]["ai_coding_innovations"],
            copyright=p["topic_distribution"]["ai_copyright_theft"],
            surv=p["topic_distribution"]["ai_mass_surveillance"],
        )
        for pid, p in USER_PROFILES.items()
    )

    marker = "## English users (synthetic)"
    section = (
        f"{marker}\n\n"
        "Labeled English users for **recovering interest profiles from posts**.\n\n"
        "Each user has a predefined `profile_id` and ground-truth "
        "`topic_distribution`. Their posts are sampled from that distribution "
        f"({POSTS_PER_USER} posts / user).\n\n"
        "Regenerate with:\n\n"
        "```bash\n"
        "python3 scripts/generate_english_users.py\n"
        "```\n\n"
        "### Scale\n\n"
        f"- **{n_users}** users (`language=en`)\n"
        f"- **{POSTS_PER_USER}** posts per user → **{n_posts}** posts\n"
        f"- **{len(USER_PROFILES)}** interest profiles (round-robin assignment)\n\n"
        "### Profiles (ground truth)\n\n"
        "| profile_id | label | P(coding / copyright / surveillance) |\n"
        "| --- | --- | --- |\n"
        f"{profile_rows}\n\n"
        "### Files\n\n"
        "| File | Format |\n| --- | --- |\n"
        "| `user_profiles.json` | Profile catalog + distributions |\n"
        "| `users.json` / `users.jsonl` / `users.csv` | Users with profile + GT dist + empirical counts |\n"
        "| `user_posts.jsonl` / `user_posts.csv` | Flat posts (`user_id`, `profile_id`, `topic`, `text`) |\n"
        "| `user_posts.json` | Aggregate metadata only (not nested texts) |\n\n"
        "### User schema (key fields)\n\n"
        "- `profile_id` / `profile_label` — discrete interest type\n"
        "- `topic_distribution` — generative ground truth over topics\n"
        "- `topic_counts` / `empirical_topic_distribution` — realized post mix\n\n"
        "### Intended use\n\n"
        "Embed or model each user's posts, estimate a topic mixture, and compare "
        "to `topic_distribution` / `profile_id` (heavy vs balanced vs split).\n"
    )

    if marker in existing:
        head, _sep, _rest = existing.partition(marker)
        existing = head.rstrip() + "\n\n" + section
    else:
        existing = existing.rstrip() + "\n\n" + section

    readme_path.write_text(existing, encoding="utf-8")
    print(f"Updated {readme_path}")


def main() -> None:
    validate_profiles()
    users = generate_users(N_USERS)
    posts = generate_posts(users)

    if len(posts) != N_USERS * POSTS_PER_USER:
        raise SystemExit(f"Expected {N_USERS * POSTS_PER_USER} posts, got {len(posts)}")
    for user in users:
        if sum(user["topic_counts"].values()) != POSTS_PER_USER:
            raise SystemExit(f"{user['user_id']}: topic_counts do not sum to {POSTS_PER_USER}")
        if user["profile_id"] not in USER_PROFILES:
            raise SystemExit(f"{user['user_id']}: unknown profile")

    write_profiles()
    write_users(users)
    write_user_posts(posts, n_users=len(users))
    update_data_readme(n_users=len(users), n_posts=len(posts))

    profile_counts = Counter(u["profile_id"] for u in users)
    print("\nUsers per profile:")
    for pid in PROFILE_IDS:
        print(f"  {pid}: {profile_counts[pid]}")
    print(f"\nDone. {len(users)} English users, {len(posts)} posts.")


if __name__ == "__main__":
    main()
