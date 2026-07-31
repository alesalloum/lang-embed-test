#!/usr/bin/env python3
"""Generate phenomenon-polarity posts about AI datacenters.

Unlike the claim–stance set (attitude *toward a claim*), this set labels
**polarity toward the phenomenon** of AI datacenter expansion:

  - pro      — favors / celebrates more AI datacenters
  - against  — opposes / criticizes AI datacenter expansion
  - neutral  — undecided, balanced, or non-committal

Scale: 300 posts × 3 polarities = 900 English texts, balanced across
six topical aspects.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TOPIC = "ai_datacenters"
TOPIC_LABEL = "Increasing number of AI datacenters"
LANGUAGE = "en"
LANGUAGE_NAME = "English"
N_PER_POLARITY = 300
SEED = 42

POLARITIES = {
    "pro": {
        "label": "Pro",
        "description": (
            "Favors AI datacenter expansion; frames buildout as beneficial, "
            "necessary, or worth supporting."
        ),
    },
    "against": {
        "label": "Against",
        "description": (
            "Opposes AI datacenter expansion; frames buildout as harmful, "
            "unnecessary, or worth resisting."
        ),
    },
    "neutral": {
        "label": "Neutral",
        "description": (
            "Does not take a side on whether AI datacenters should expand; "
            "hedges, asks questions, or presents tradeoffs without a verdict."
        ),
    },
}

ASPECTS = {
    "economic": {
        "label": "Economic",
        "description": "Jobs, investment, taxes, energy prices, regional growth, capital markets.",
    },
    "environmental": {
        "label": "Environmental",
        "description": "Energy use, water, emissions, land use, renewables, climate tradeoffs.",
    },
    "infrastructure": {
        "label": "Infrastructure",
        "description": "Power grids, transmission, cooling, fiber, roads, construction capacity.",
    },
    "geopolitical": {
        "label": "Geopolitical",
        "description": "National security, chip supply, sovereignty, alliances, export controls.",
    },
    "local_community": {
        "label": "Local community",
        "description": "Zoning, noise, housing, schools, community benefits, local politics.",
    },
    "technological": {
        "label": "Technological",
        "description": "GPU clusters, cooling tech, efficiency, networking, model training scale.",
    },
}

PLACES = [
    "northern Virginia", "central Texas", "rural Iowa", "Arizona's desert corridor",
    "eastern Oregon", "upstate New York", "Georgia's coastal plain", "Ohio's Rust Belt towns",
    "Quebec's hydropower belt", "Ireland's midlands", "Singapore's industrial parks",
    "South Korea's semiconductor belt", "the UAE's free zones", "Chile's Atacama edge",
    "Finland's cold-climate parks", "Spain's solar belt", "Poland's logistics corridor",
    "Malaysia's Johor corridor", "India's Hyderabad tech belt", "Brazil's São Paulo hinterland",
    "Canada's prairie provinces", "the Nordics", "Gulf Coast industrial zones",
    "the Pacific Northwest", "Midwest farm counties", "suburban Phoenix",
    "northern Italy's industrial belt", "Japan's Tohoku rebuild zones",
    "Australia's renewable zones", "Morocco's Atlantic coast",
]

ACTORS = [
    "hyperscalers", "AI labs", "cloud providers", "chipmakers", "utility companies",
    "private equity funds", "state investment boards", "municipal governments",
    "regional economic-development agencies", "sovereign wealth funds",
    "colocation operators", "telecom carriers", "energy traders",
    "construction consortia", "national labs", "defense contractors",
    "startup GPU cloud firms", "REIT-backed campus developers",
]

FACILITIES = [
    "AI training campuses", "inference megacampuses", "GPU-dense halls",
    "liquid-cooled clusters", "multi-gigawatt campuses", "edge AI pods",
    "sovereign compute hubs", "research supercomputer halls",
    "modular container datacenters", "subsea-cable landing campuses",
    "nuclear-adjacent compute parks", "behind-the-meter solar campuses",
    "black-start resilient bunkers", "waste-heat district heating sites",
]

# Aspect-specific argument cues (valence assigned by polarity templates)
PRO_CUES = {
    "economic": [
        "anchor high-wage construction and ops jobs",
        "broaden the local tax base without raising rates",
        "pull private capital into overlooked counties",
        "create supplier ecosystems around each campus",
        "stabilize municipal budgets after plant closures",
        "turn idle industrial land into productive assets",
    ],
    "environmental": [
        "pair new load with additionality-backed renewables",
        "fund grid upgrades that cut curtailment waste",
        "reuse waste heat for district heating",
        "accelerate nuclear and geothermal firming builds",
        "retire dirtier peaker plants as campuses modernize",
        "push cooling R&D that lowers water intensity",
    ],
    "infrastructure": [
        "force long-delayed transmission upgrades",
        "modernize substations communities already needed",
        "underwrite fiber laterals to rural towns",
        "pay for resilient black-start capacity",
        "co-fund roads and water mains with developers",
        "seed modular generation that firms local grids",
    ],
    "geopolitical": [
        "keep frontier training capacity on allied soil",
        "reduce dependence on overseas inference supply",
        "give democracies leverage in model-access talks",
        "secure chip-to-cloud pipelines domestically",
        "support sovereign compute for public research",
        "deter adversaries by scaling trusted infrastructure",
    ],
    "local_community": [
        "negotiate real community-benefit agreements",
        "fund scholarships tied to campus payrolls",
        "convert empty retail pads into training centers",
        "bring broadband as a side effect of fiber builds",
        "give towns a seat at siting negotiations",
        "replace dying extractive payrolls with steadier ones",
    ],
    "technological": [
        "unlock larger training runs that advance open science",
        "drive liquid-cooling efficiency gains industry-wide",
        "let startups rent frontier GPUs without owning fabs",
        "push networking standards that cut idle capacity",
        "make inference cheaper for hospitals and labs",
        "concentrate R&D where talent and power already meet",
    ],
}

AGAINST_CUES = {
    "economic": [
        "inflate industrial power rates for everyone else",
        "capture tax abatements while creating few permanent jobs",
        "crowd out manufacturing that employs denser workforces",
        "leave towns holding stranded-asset risk after the boom",
        "concentrate gains in remote shareholders, not locals",
        "bid up housing costs faster than wages rise",
    ],
    "environmental": [
        "drain aquifers in drought-stressed basins",
        "lock in gas peakers for 'temporary' firming",
        "convert habitat into sealed concrete campuses",
        "spike Scope-2 emissions despite renewable marketing",
        "dump diesel NOx during grid emergencies",
        "treat water as an infinite free input",
    ],
    "infrastructure": [
        "queue-jump households waiting for grid interconnects",
        "overload aging transformers and feeders",
        "force ratepayers to bankroll private transmission",
        "consume cooling water meant for farms and towns",
        "clog roads with construction traffic for years",
        "starve public projects of skilled trades labor",
    ],
    "geopolitical": [
        "centralize strategic compute in a few corporate hands",
        "invite sanctions exposure through opaque ownership",
        "export control workarounds via third-country campuses",
        "militarize civilian grids under dual-use cover",
        "undermine alliances when water and power fights spill over",
        "race capacity without democratic oversight",
    ],
    "local_community": [
        "steamroll zoning with state preemption bills",
        "add continuous industrial noise next to homes",
        "buy silence with scholarships while rates climb",
        "hollow out civic life into permit fights",
        "turn schools into bargaining chips for abatements",
        "leave neighbors with light pollution and truck traffic",
    ],
    "technological": [
        "overbuild GPU halls that sit half-idle after the hype",
        "optimize for closed models instead of public compute",
        "waste cooling capacity on vanity cluster sizes",
        "lock buyers into proprietary interconnect stacks",
        "chase density that outruns safety and maintainability",
        "treat efficiency claims as marketing, not measured gains",
    ],
}

NEUTRAL_CUES = {
    "economic": [
        "job quality versus abatement cost",
        "whether tax base gains outlast construction payrolls",
        "local multipliers versus shareholder leakage",
        "energy-price pass-through to smaller firms",
        "boom-town risk if a single tenant leaves",
        "how permanent the ops headcount really is",
    ],
    "environmental": [
        "renewable additionality versus paper PPAs",
        "water budgets under multi-year drought",
        "waste-heat reuse that actually materializes",
        "peaker plants sold as 'bridge' capacity",
        "land-use tradeoffs against habitat corridors",
        "lifecycle emissions beyond campus fences",
    ],
    "infrastructure": [
        "who pays for interconnect upgrades",
        "timeline realism for new transmission",
        "cooling-water competing with farm allotments",
        "whether fiber laterals reach homes or only campuses",
        "construction labor shortages cascading to hospitals",
        "grid resilience claims versus measured outages",
    ],
    "geopolitical": [
        "sovereign compute versus corporate capture",
        "alliance benefits versus local resource strain",
        "export-control exposure in cross-border training",
        "how much capacity is dual-use in practice",
        "domestic buildouts that still depend on foreign chips",
        "transparency of ownership in free-zone campuses",
    ],
    "local_community": [
        "community benefits that survive the press release",
        "noise and traffic mitigation that is enforceable",
        "housing supply responses to new payrolls",
        "school funding versus industrial rate pressure",
        "whether locals get real veto power on siting",
        "civic trust after rushed permit processes",
    ],
    "technological": [
        "utilization rates after the first training boom",
        "open-access quotas versus reserved enterprise capacity",
        "cooling-efficiency claims under real weather",
        "interoperability across vendors",
        "safety margins at extreme rack densities",
        "whether smaller distributed pods beat megacampuses",
    ],
}

PRO_OPENERS = [
    "I'm for more AI datacenters.",
    "Build the campuses.",
    "Say it plainly: we need the capacity.",
    "Pro-expansion here.",
    "Count me as a yes on the buildout.",
    "This wave is worth backing.",
    "Greenlight the campuses.",
    "I'm bullish on more AI datacenters.",
]

AGAINST_OPENERS = [
    "I'm against the AI datacenter surge.",
    "Stop rubber-stamping these campuses.",
    "Hard no on endless AI datacenter expansion.",
    "We should block this buildout.",
    "Against more AI datacenters in my book.",
    "This expansion is a bad deal.",
    "Hit the brakes on new campuses.",
    "I oppose the AI datacenter boom.",
]

NEUTRAL_OPENERS = [
    "Still undecided on the AI datacenter wave.",
    "Not picking a side on more campuses yet.",
    "Holding judgment on AI datacenter expansion.",
    "The buildout debate isn't settled for me.",
    "Watching the AI datacenter fight without a banner.",
    "Neither booster nor blocker — still weighing it.",
    "Open question: should we keep adding campuses?",
    "Parking the yes/no on AI datacenters for now.",
]

PRO_CLOSERS = [
    "Approve the next round.",
    "Expansion is the right call.",
    "Let's build.",
    "Capacity is policy.",
    "I'm here for the buildout.",
]

AGAINST_CLOSERS = [
    "Reject the next campus.",
    "Enough is enough.",
    "Protect the commons first.",
    "No more blank checks.",
    "Opposition is the responsible stance.",
]

NEUTRAL_CLOSERS = [
    "Show me cleaner evidence either way.",
    "I need receipts before choosing.",
    "Tradeoffs first, slogans later.",
    "Still collecting local data.",
    "Ask me again after the next rate case.",
]


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" .", ".").replace(" ,", ",")
    return s


def _post_id(polarity: str, idx: int) -> str:
    return f"{TOPIC}_polarity_{polarity}_{idx:03d}"


def _hash_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _balanced_aspects(n: int, rng: random.Random) -> list[str]:
    keys = list(ASPECTS.keys())
    base, rem = divmod(n, len(keys))
    out: list[str] = []
    for i, k in enumerate(keys):
        out.extend([k] * (base + (1 if i < rem else 0)))
    rng.shuffle(out)
    return out


def _pro_text(aspect: str, place: str, actors: str, facilities: str, cue: str, rng: random.Random) -> str:
    templates = [
        (
            f"{rng.choice(PRO_OPENERS)} In {place}, more {facilities} from {actors} can "
            f"{cue}. That is why I support the expansion. {rng.choice(PRO_CLOSERS)}"
        ),
        (
            f"{rng.choice(PRO_OPENERS)} The case in {place} is straightforward: {actors}' "
            f"{facilities} help {cue}. AI datacenter growth is good policy here. "
            f"{rng.choice(PRO_CLOSERS)}"
        ),
        (
            f"Pro take on {place}: back the {facilities}. When {actors} scale responsibly, "
            f"they {cue}. I'm for more AI datacenters. {rng.choice(PRO_CLOSERS)}"
        ),
        (
            f"{rng.choice(PRO_OPENERS)} Don't slow-walk {facilities} around {place}. "
            f"Done right by {actors}, the buildout will {cue}. {rng.choice(PRO_CLOSERS)}"
        ),
    ]
    return _clean(rng.choice(templates))


def _against_text(aspect: str, place: str, actors: str, facilities: str, cue: str, rng: random.Random) -> str:
    actors_cap = actors[0].upper() + actors[1:]
    templates = [
        (
            f"{rng.choice(AGAINST_OPENERS)} In {place}, more {facilities} from {actors} "
            f"mostly {cue}. That is a reason to resist the boom. {rng.choice(AGAINST_CLOSERS)}"
        ),
        (
            f"{rng.choice(AGAINST_OPENERS)} The {place} story is clear enough: {actors}' "
            f"{facilities} tend to {cue}. I don't want more AI datacenters. "
            f"{rng.choice(AGAINST_CLOSERS)}"
        ),
        (
            f"Against take on {place}: block the next {facilities}. {actors_cap} keep "
            f"selling growth while they {cue}. {rng.choice(AGAINST_CLOSERS)}"
        ),
        (
            f"{rng.choice(AGAINST_OPENERS)} Slowing {facilities} near {place} is justified "
            f"when {actors} {cue}. {rng.choice(AGAINST_CLOSERS)}"
        ),
    ]
    return _clean(rng.choice(templates))


def _neutral_text(aspect: str, place: str, actors: str, facilities: str, cue: str, rng: random.Random) -> str:
    templates = [
        (
            f"{rng.choice(NEUTRAL_OPENERS)} On {place}, people argue about whether "
            f"{actors}' {facilities} help or hurt — especially {cue}. "
            f"{rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(NEUTRAL_OPENERS)} Flagging {facilities} in {place} without a "
            f"verdict. The live issue is {cue}. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"Neutral note on {place}: if {actors} add more {facilities}, I'm watching "
            f"{cue} before I cheer or protest. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
        (
            f"{rng.choice(NEUTRAL_OPENERS)} Re: {facilities} around {place} — boosters and "
            f"critics both cite {cue}. No banner from me yet. {rng.choice(NEUTRAL_CLOSERS)}"
        ),
    ]
    return _clean(rng.choice(templates))


def generate_records(rng: random.Random) -> list[dict]:
    records: list[dict] = []
    seen_text: set[str] = set()
    builders = {
        "pro": (_pro_text, PRO_CUES),
        "against": (_against_text, AGAINST_CUES),
        "neutral": (_neutral_text, NEUTRAL_CUES),
    }

    for polarity, (builder, cue_bank) in builders.items():
        aspects = _balanced_aspects(N_PER_POLARITY, rng)
        for i, aspect in enumerate(aspects):
            # Retry until unique text
            text = ""
            place = actors = facilities = cue = ""
            for _attempt in range(80):
                place = rng.choice(PLACES)
                actors = rng.choice(ACTORS)
                facilities = rng.choice(FACILITIES)
                cue = rng.choice(cue_bank[aspect])
                text = builder(aspect, place, actors, facilities, cue, rng)
                key = text.lower()
                if key not in seen_text and len(text) >= 60:
                    seen_text.add(key)
                    break
            else:
                # Extremely unlikely; force uniqueness with hash salt
                text = _clean(f"{text} [{_hash_key(polarity, aspect, str(i))}]")
                seen_text.add(text.lower())

            records.append(
                {
                    "post_id": _post_id(polarity, i),
                    "topic": TOPIC,
                    "topic_label": TOPIC_LABEL,
                    "aspect": aspect,
                    "aspect_label": ASPECTS[aspect]["label"],
                    "polarity": polarity,
                    "polarity_label": POLARITIES[polarity]["label"],
                    "place": place,
                    "actors": actors,
                    "facilities": facilities,
                    "cue": cue,
                    "language": LANGUAGE,
                    "language_name": LANGUAGE_NAME,
                    "text": text,
                }
            )
    rng.shuffle(records)
    # Stable index after shuffle for embedding row alignment notes
    for idx, rec in enumerate(records):
        rec["index"] = idx
    return records


def write_outputs(records: list[dict]) -> Path:
    out_dir = DATA_DIR / "polarity_posts"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "posts.json"
    jsonl_path = out_dir / "posts.jsonl"
    csv_path = out_dir / "posts.csv"

    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fieldnames = [
        "index",
        "post_id",
        "topic",
        "topic_label",
        "aspect",
        "aspect_label",
        "polarity",
        "polarity_label",
        "place",
        "actors",
        "facilities",
        "cue",
        "language",
        "language_name",
        "text",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    counts = {p: sum(1 for r in records if r["polarity"] == p) for p in POLARITIES}
    aspect_counts = {
        a: sum(1 for r in records if r["aspect"] == a) for a in ASPECTS
    }
    polarity_aspect = {
        p: {a: sum(1 for r in records if r["polarity"] == p and r["aspect"] == a) for a in ASPECTS}
        for p in POLARITIES
    }

    summary = {
        "topic": TOPIC,
        "topic_label": TOPIC_LABEL,
        "language": LANGUAGE,
        "polarities": list(POLARITIES.keys()),
        "aspects": list(ASPECTS.keys()),
        "counts": {
            "total": len(records),
            "per_polarity": counts,
            "aspect_counts": aspect_counts,
            "polarity_aspect": polarity_aspect,
        },
        "files": {
            "posts.json": "Array of polarity posts",
            "posts.jsonl": "One JSON object per post",
            "posts.csv": "Flat CSV with the same columns",
        },
        "labeling_notes": (
            "Ground-truth label is `polarity` toward the phenomenon of AI "
            "datacenter expansion (pro / against / neutral), NOT agreement "
            "with a specific claim. Aspect is a topical facet used for "
            "secondary encoding in plots."
        ),
        "seed": SEED,
        "n_per_polarity": N_PER_POLARITY,
    }
    (out_dir / "info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# AI datacenter phenomenon-polarity posts",
                "",
                "Synthetic social-media posts labeled by **polarity toward AI",
                "datacenter expansion** (not claim-agreement stance).",
                "",
                "## Topic",
                "",
                f"- `{TOPIC}`: {TOPIC_LABEL}",
                "",
                "## Polarities",
                "",
                *[
                    f"- `{k}`: {v['description']}"
                    for k, v in POLARITIES.items()
                ],
                "",
                "## Aspects",
                "",
                *[
                    f"- `{k}`: {v['label']} — {v['description']}"
                    for k, v in ASPECTS.items()
                ],
                "",
                "## Scale",
                "",
                f"- **{N_PER_POLARITY}** posts × **3** polarities = **{len(records)}** texts",
                f"- English only (`language={LANGUAGE}`)",
                f"- Aspect mix: {', '.join(f'{k}={v}' for k, v in aspect_counts.items())}",
                "",
                "## Files",
                "",
                "| File | Format |",
                "| --- | --- |",
                "| `posts.json` | Array of posts |",
                "| `posts.jsonl` | One object per post |",
                "| `posts.csv` | Flat CSV |",
                "| `info.json` | Counts + labeling notes |",
                "",
                "## How it was built",
                "",
                "```bash",
                "python3 scripts/generate_ai_datacenter_polarity.py",
                "```",
                "",
                "## Schema (key fields)",
                "",
                "- `post_id` — unique id",
                "- `polarity` — `pro` | `against` | `neutral`",
                "- `aspect` / `aspect_label` — topical facet",
                "- `text` — social-media post expressing that polarity",
                "",
                "## Intended use",
                "",
                "Embed all posts and test whether vectors separate by `polarity`",
                "toward AI datacenter expansion (vanilla vs polarity-instruct).",
                "",
                "```json",
                json.dumps(summary, indent=2),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_dir


def update_data_readme(out_dir: Path) -> None:
    readme = DATA_DIR / "README.md"
    marker = "## AI datacenter phenomenon-polarity set"
    section = """## AI datacenter phenomenon-polarity set

Posts labeled **pro / against / neutral toward AI datacenter expansion**
(phenomenon polarity, not claim-agreement).

See [`polarity_posts/`](polarity_posts/).

```bash
python3 scripts/generate_ai_datacenter_polarity.py
```

- **300** posts × **3** polarities = **900** English posts
- Ground-truth label: `polarity`; secondary facet: `aspect`
"""
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    if marker in text:
        # Replace existing section through next ## or EOF
        parts = text.split(marker)
        rest = parts[1]
        next_h = rest.find("\n## ")
        if next_h >= 0:
            text = parts[0] + section + rest[next_h + 1 :]
        else:
            text = parts[0] + section
    else:
        text = text.rstrip() + "\n\n" + section
    readme.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def update_root_readme() -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    bullet = (
        "- **AI datacenter polarity set**: 300 pro / 300 neutral / 300 against "
        "under [`data/polarity_posts/`](data/polarity_posts/)"
    )
    if "polarity_posts" not in text:
        needle = (
            "- **AI datacenter claim–stance set**: 500 claims × 3 stances "
            "(supportive/neutral/critical) under [`data/claims_stances/`](data/claims_stances/)"
        )
        if needle in text:
            text = text.replace(needle, needle + "\n" + bullet)
        else:
            text = text.rstrip() + "\n" + bullet + "\n"

    regen = (
        "\nRegenerate AI datacenter polarity posts with:\n\n"
        "```bash\n"
        "python3 scripts/generate_ai_datacenter_polarity.py\n"
        "```\n"
    )
    if "generate_ai_datacenter_polarity.py" not in text:
        anchor = "python3 scripts/generate_ai_datacenter_claims.py\n```"
        if anchor in text:
            text = text.replace(
                anchor,
                anchor
                + "\n\nRegenerate AI datacenter polarity posts with:\n\n"
                "```bash\n"
                "python3 scripts/generate_ai_datacenter_polarity.py\n"
                "```",
            )
        else:
            text = text.rstrip() + regen

    emb_blurb = (
        "\nPolarity posts are embedded twice under "
        "[`data/embeddings/polarity_qwen3-embedding-0.6b/`]"
        "(data/embeddings/polarity_qwen3-embedding-0.6b/):\n\n"
        "- `vanilla/` — no instruction\n"
        "- `polarity_instruct/` — Instruct/Query prompt asking the model to\n"
        "  encode pro / against / neutral polarity toward AI datacenters\n\n"
        "```bash\n"
        "python scripts/embed_polarity.py\n"
        "```\n"
    )
    if "embed_polarity.py" not in text:
        anchor = "python scripts/embed_claim_stances.py\n```"
        if anchor in text:
            text = text.replace(anchor, anchor + emb_blurb)
        else:
            text = text.rstrip() + emb_blurb

    plot_blurb = (
        "\nPolarity UMAP (polarity = color, aspect = marker shape; vanilla vs\n"
        "polarity-instruct comparison) lands in\n"
        "[`results/polarity_umap/`](results/polarity_umap/):\n\n"
        "```bash\n"
        "python scripts/plot_polarity_umap.py\n"
        "```\n"
    )
    if "plot_polarity_umap.py" not in text:
        anchor = "python scripts/plot_claim_stance_umap.py\n```"
        if anchor in text:
            text = text.replace(anchor, anchor + plot_blurb)
        else:
            text = text.rstrip() + plot_blurb

    readme.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    records = generate_records(rng)
    assert len(records) == N_PER_POLARITY * 3
    for p in POLARITIES:
        assert sum(1 for r in records if r["polarity"] == p) == N_PER_POLARITY
    texts = [r["text"] for r in records]
    assert len(texts) == len(set(t.lower() for t in texts))
    out_dir = write_outputs(records)
    update_data_readme(out_dir)
    update_root_readme()
    print(f"Wrote {len(records)} polarity posts -> {out_dir}")


if __name__ == "__main__":
    main()
