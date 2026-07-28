#!/usr/bin/env python3
"""Generate multilingual toy social-media posts for embedding/clustering tests.

Three topics × 50 English posts, each translated to Arabic, Spanish, and Chinese.
Posts that share the same meaning keep a shared `post_id` across languages so
you can check whether k-means clusters by topic (desired) vs language (failure).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TOPICS = {
    "ai_coding_innovations": {
        "label": "AI innovations in coding",
        "description": "Posts about AI tools improving software development, coding assistants, and engineering productivity.",
    },
    "ai_copyright_theft": {
        "label": "AI theft of artistic and authors' copyrights",
        "description": "Posts about generative AI copying artists/writers without consent or compensation.",
    },
    "ai_mass_surveillance": {
        "label": "AI-enabled mass surveillance",
        "description": "Posts about facial recognition, predictive policing, and AI monitoring of populations.",
    },
}

LANGUAGES = {
    "en": "English",
    "ar": "Arabic",
    "es": "Spanish",
    "zh": "Chinese (Simplified)",
}

# ---------------------------------------------------------------------------
# English source posts (50 per topic). Imaginary social-media style posts.
# ---------------------------------------------------------------------------

ENGLISH_POSTS: dict[str, list[str]] = {
    "ai_coding_innovations": [
        "Just shipped a feature in half the time thanks to my AI coding assistant. The autocomplete actually understood my intent this sprint.",
        "Hot take: AI pair programmers won't replace developers, but developers who ignore them will get left behind.",
        "Spent the morning refactoring legacy Python with Copilot-style suggestions. It caught three edge cases I would have missed.",
        "Our team cut PR review time by 40% after adding an AI lint that explains *why* a change is risky.",
        "Tried generating unit tests with an LLM today. Coverage jumped from 62% to 89% before lunch.",
        "The new AI debugger in my IDE traced a race condition across three services. Felt like science fiction.",
        "Prompting an AI to draft SQL migrations saved me from a midnight outage. Still reviewed every line, but wow.",
        "Junior eng on my team used an AI tutor to learn Rust concurrency. They're shipping production code in weeks, not months.",
        "AI-assisted code search found the exact helper function buried in a monorepo of 2M lines. Search is finally useful.",
        "We're experimenting with AI-generated API docs from OpenAPI specs. Onboarding new hires is dramatically smoother.",
        "Natural language to Terraform is getting scary good. I described a VPC layout and got a solid first draft.",
        "My terminal now has an AI that explains cryptic compiler errors in plain English. Peak 2020s developer experience.",
        "Used an AI to migrate a React class component mess to hooks. It preserved behavior and cleaned naming conventions.",
        "The best coding AIs aren't the ones that write the most code — they're the ones that ask clarifying questions first.",
        "Automated dependency upgrades via AI PRs: 18 merged this week, zero rollbacks. Cautiously optimistic.",
        "Voice-to-code demos still feel gimmicky, but dictating a regex and getting a correct one back? I'll take it.",
        "Our CI now runs an AI flaky-test analyzer. It clustered failures by root cause instead of blaming 'randomness'.",
        "Sketching architecture on a whiteboard, photographing it, and getting a sequence diagram back from AI. Wild workflow.",
        "I asked an AI to explain our auth middleware like I was new. It was clearer than our wiki. Embarrassing and useful.",
        "Code review bots that suggest alternative algorithms with Big-O tradeoffs are changing how we mentor juniors.",
        "Shipped a CLI tool where half the boilerplate came from an AI scaffold. I focused on the domain logic instead.",
        "Multimodal coding assistants that read screenshots of UI bugs and propose CSS fixes are underrated right now.",
        "We fine-tuned a small model on our internal style guide. Diff noise in AI suggestions dropped overnight.",
        "AI filled in the GraphQL resolvers from our schema comments. Not perfect, but a fantastic starting point.",
        "The future of coding is collaborative: human judgment plus AI speed. Today felt like that future arrived.",
        "Translated a 10k-line Java module to Go with AI help. Manual review took longer than generation — as it should.",
        "An AI suggested memoizing a hot path I had never profiled. Benchmarks confirmed a 3× win. Humble pie.",
        "Pairing with an AI on LeetCode-style interview prep is weirdly effective. It forces you to verbalize tradeoffs.",
        "Our design system now has an AI that proposes accessible ARIA attributes when you paste a component.",
        "Generated a fuzzing harness with an LLM and it found a buffer edge case within an hour. Security loves this.",
        "I'm less afraid of blank files now. AI drafts give me something to react to instead of staring at the void.",
        "Real-time collaborative coding with an AI observer that flags security smells mid-session. Instant feedback loop.",
        "Asked the model to rewrite our Makefile for clarity. Developers stopped pinging me about build targets.",
        "AI-assisted git bisect summaries are saving hours when hunting regressions across long commit histories.",
        "The coding agent booked a calendar hold, opened a draft PR, and left a checklist. I just approved the plan.",
        "Using AI to convert Figma annotations into typed props for our design components. Design-dev handoff improved.",
        "I still write critical path crypto by hand. Everything else? Happy to let the assistant draft and me verify.",
        "Benchmarking three AI coding tools on the same bugfix. Clear winner on multi-file reasoning this week.",
        "Natural language queries over our codebase ('where do we validate JWTs?') beat tribal knowledge every time.",
        "An AI suggested deleting 400 lines of dead feature-flag code. Ship it. Less code is a gift.",
        "Teaching the intern: use AI for boilerplate, use your brain for invariants. That rule is sticking.",
        "Our incident bot now proposes rollback commands from stack traces. Mean time to mitigate is down again.",
        "Generated end-to-end Playwright tests from a user story. Flaky in places, but the skeleton was solid.",
        "I love that coding AIs can explain unfamiliar DSLs. Picked up a new build system without a week of docs.",
        "AI-written commit messages that reference the ticket and summarize intent — finally consistent history.",
        "Tried agentic coding for a weekend side project. It handled scaffolding; I handled taste and product decisions.",
        "Static analysis plus LLM explanations means juniors understand *why* a rule exists, not just that it failed.",
        "Ported our logging format across five services with an AI batch edit. Consistency without the tedious grind.",
        "The best demo I saw this month: describe a bug in Slack, get a candidate patch in the thread. Closing the loop.",
        "Cautious optimism: AI coding tools amplify careful engineers. They also amplify messy habits. Discipline still wins.",
    ],
    "ai_copyright_theft": [
        "Artists spent years developing a style. AI scraped it overnight without consent or credit. That's not innovation — that's theft.",
        "My novel is in training corpora and I never opted in. Publishers need to fight this harder.",
        "Stock photo sites are flooded with AI clones of living illustrators. Clients can't tell; creators get unpaid.",
        "If a model can pastiche my brushwork, it learned from my portfolio. Pay us or don't train on us.",
        "Copyright offices are scrambling. Meanwhile generators keep shipping 'in the style of' prompts as a feature.",
        "Writers: check if your books appear in shadow libraries used for training. Mine did. Furious doesn't cover it.",
        "Commissioning an illustrator used to mean supporting a human. Now brands prompt and call it 'efficiency'.",
        "AI music that mimics a living artist's voice without a license should be illegal, full stop.",
        "I watermarked my art. Scrapers still vacuumed it. Watermarks aren't consent.",
        "Studios train on fan art and indie comics, then compete with the same communities. Predatory loop.",
        "Fair use was never meant to cover industrial-scale ingestion of entire libraries. Courts need to say so.",
        "My agent found AI summaries of my articles sold as 'original research'. Attribution is broken online.",
        "Photographers losing stock licenses to synthetic lookalikes. The market is being hollowed out.",
        "Opt-out registries are a joke when models already trained. We need opt-in and compensation, not paperwork theater.",
        "Calling it 'inspiration' when the model regurgitates a signature character design is gaslighting creators.",
        "Game concept artists report briefs that literally say 'make it like this AI mashup of our unpaid references'.",
        "Authors Guild lawsuits matter. Individual freelancers can't litigate against trillion-parameter scrapers alone.",
        "I spent a decade building an audience. Generators freeride on that recognition without hiring me once.",
        "Translate 'style transfer' into honest language: unauthorized derivative works at planetary scale.",
        "Museums digitizing collections then licensing them to AI firms — did the original artists agree?",
        "Voice actors clone-scammed by AI reads of their demos. Consent forms should be mandatory before synthesis.",
        "Fanfic writers joked about copyright; now corporate AIs treat all fiction as free fuel. Irony is bitter.",
        "Design agencies pitching 'AI-accelerated' decks built on stolen moodboards. Clients deserve transparency.",
        "If training data is the new oil, then unpaid creators are the unpaid oilfield workers. That has to change.",
        "My comic panels showed up in a dataset dump. No license, no email, no revenue share. Just extraction.",
        "Synthetic 'authors' flooding ebook stores with AI sludge. Discoverability for real writers collapses.",
        "Fashion designers seeing AI knockoffs of unreleased sketches after studio leaks. IP law is lagging hard.",
        "Open-source models trained on pirated books aren't 'open' — they're laundering copyrighted text.",
        "I support AI tools that license datasets fairly. I oppose tools that treat culture as an all-you-can-eat buffet.",
        "Prompt markets selling 'exact artist style' packs should be sued into oblivion.",
        "Journalists finding their paywalled reporting inside chatbot answers with zero outbound links. Theft with a UI.",
        "Children's book illustrators undercut by same-day AI picture books. Quality aside, the labor theft is real.",
        "Collective licensing could work — if platforms admit they need it instead of hiding behind 'publicly available'.",
        "My Patreon exclusives were scraped from a leak. Generators don't care about paywalls or trust.",
        "Calling artists Luddites for wanting payment is a tactic. We use tools; we refuse unpaid appropriation.",
        "Film concept art leaks into training sets before the movie releases. Spoilers and IP theft in one package.",
        "Academic publishers quietly selling corpora to AI companies while authors see none of the upside.",
        "I can spot my linework in 'original' NFT drops. Blockchain doesn't clean dirty training data.",
        "Songwriters: your melodies are being statistically averaged into royalty-free sludge. Fight for mechanicals.",
        "Translation rights matter too. AI translating my novel without a deal still violates my copyright.",
        "Studios using AI to replace background artists after training on those same artists' years of work. Cruel.",
        "We need provenance standards: every generative output should disclose training sources when feasible.",
        "Creative unions organizing around AI consent are the only reason some platforms even mention compensation.",
        "'Publicly scraped' is not a moral license. Public park ≠ free to bulldoze for a data center.",
        "My workshop slides ended up in a commercial coding-model corpus. Teaching materials aren't free training fuel.",
        "Indie game pixel artists seeing asset packs that clone their palettes and animations. Death by a thousand gens.",
        "Until models can prove clean data lineage, I assume every 'free' generator is built on unpaid labor.",
        "Editors cutting human cover artists because AI is 'good enough'. Good enough built on whose portfolios?",
        "Copyright maximalism isn't the goal — consent, credit, and compensation are. AI broke all three.",
        "If your demo needs 'in the style of living artist X', you already know you're crossing an ethical line.",
    ],
    "ai_mass_surveillance": [
        "Cities deploying real-time facial recognition on every corner. Dissent becomes a searchable database.",
        "Predictive policing models recycle historical bias and call it objective risk scoring. Communities notice.",
        "Your smart doorbell footage is feeding vendor models. Convenience is the recruitment pitch for surveillance.",
        "AI that flags 'suspicious' gait patterns in malls should terrify anyone who values anonymity in public.",
        "Governments buying emotion-recognition cameras for protests. Pseudoscience with handcuffs attached.",
        "License plate readers plus cloud AI means your daily routes are a permanent dossier. Opting out isn't offered.",
        "Schools scanning student faces for attendance normalize biometric tracking before kids can consent.",
        "When chat apps offer 'AI safety scanning' of private messages, assume content is classified at scale.",
        "Drone swarms with onboard person re-identification turn neighborhoods into open-air panopticons.",
        "Credit scorers experimenting with social media sentiment AI. Your jokes could price your loan.",
        "Border AI that predicts 'intent' from microexpressions is prejudice automated and exported.",
        "Employers monitoring keystroke dynamics and webcam attention scores. Productivity theater meets dystopia.",
        "Public CCTV upgraded with AI search: find everyone who wore a red jacket near the plaza. Chilling effect incoming.",
        "Voice assistants always listening for a wake word still buffer audio. That buffer is a surveillance surface.",
        "Municipalities claim anonymized heatmaps while vendors keep re-identification keys. Trust, but verify — then don't trust.",
        "AI moderation tools used by police to scrape activist Discords. Community spaces aren't safe by default anymore.",
        "Smart streetlights that track phones via MAC randomization defeat — until AI fingerprinting catches up.",
        "Insurance apps wanting continuous driving AI scores. Refuse and pay more. Coerced surveillance.",
        "Mass scraping of social posts to build 'threat' graphs. Speech becomes evidence before any crime exists.",
        "Hospitals piloting AI visitor face logs. Medical privacy shouldn't end at the lobby camera.",
        "Autonomous cameras that zoom on 'unusual loitering' encode poverty as suspicion.",
        "National ID plus AI face match at every transit gate is a recipe for political control, not safety.",
        "Retailers using AI to score shoppers for theft risk by appearance. Discrimination with a dashboard.",
        "Encrypted messaging is under pressure precisely because it blocks wholesale AI content surveillance.",
        "City dashboards promising 'safer streets' while quietly expanding always-on audio analytics. Read the procurement docs.",
        "Refugee camps monitored by AI perimeter systems. Vulnerable people get the densest surveillance first.",
        "Your fitness tracker location trails sold into data brokers, then enriched with AI lifestyle profiles.",
        "Police bodycams with live AI prompting officers — who audits the prompts when they escalate encounters?",
        "Social credit experiments don't need that name to arrive. Behavior scores from AI cameras are enough.",
        "Airports expanding AI behavioral detection beyond security theater into permanent passenger dossiers.",
        "Landlords installing AI hallway cams that alert on 'unauthorized guests'. Housing becomes a watchlist.",
        "Open-source face datasets built without consent power commercial surveillance stacks worldwide.",
        "When every lamppost can run a person detector, freedom of assembly depends on who controls the model.",
        "AI tip lines that auto-prioritize anonymous reports will amplify harassment campaigns against minorities.",
        "Car makers shipping cabin cameras with driver-attention AI. Your face becomes telematics.",
        "Public Wi-Fi captive portals harvesting device IDs for AI foot-traffic analytics. Free internet, costly privacy.",
        "Fusion centers correlating protest livestreams with face search. Journalism and activism both at risk.",
        "Child monitoring apps marketed to parents become government-accessible telemetry with one subpoena.",
        "Stadiums scanning crowds with AI weapons detection — accuracy claims hide false positive rates on brown skin.",
        "Smart cities pitch efficiency; the architecture is continuous population instrumentation.",
        "Whistleblowers warned about bulk biometric databases years ago. Procurement just kept accelerating.",
        "AI that predicts 'radicalization' from library checkouts and search history is McCarthyism with GPUs.",
        "Delivery robots mapping sidewalks with cameras normalize private fleets as mobile surveillance nodes.",
        "Prisons using AI to score inmate 'risk' from call transcripts. Speech inside walls is never free.",
        "Corporate campuses tracking badge plus face continuous authentication. Workers are perpetual suspects.",
        "Newsrooms doxed by AI reverse image search on blurred sources. Protection of sources is eroding.",
        "Environmental sensors that also run mics for 'gunshot detection' double as protest audio collectors.",
        "Once face search is cheap, every uploaded party photo is a potential identification query against you.",
        "Democratic oversight of AI surveillance lags deployment by years. That gap is the product.",
        "I want safer cities. I don't want a searchable catalog of every face that walked past city hall.",
    ],
}


def translate_batch(texts: list[str], target: str, max_retries: int = 5) -> list[str]:
    """Translate a list of English texts to `target` language code."""
    translator = GoogleTranslator(source="en", target=target)
    out: list[str] = []
    for i, text in enumerate(texts):
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                translated = translator.translate(text)
                if not translated:
                    raise RuntimeError("empty translation")
                out.append(translated)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — network / rate limits
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise RuntimeError(f"Failed translating item {i} to {target}: {last_err}") from last_err
        if (i + 1) % 10 == 0:
            print(f"  [{target}] {i + 1}/{len(texts)}")
            time.sleep(0.4)
    return out


def build_records() -> list[dict]:
    records: list[dict] = []
    for topic_id, posts in ENGLISH_POSTS.items():
        assert len(posts) == 50, f"{topic_id} has {len(posts)} posts, expected 50"
        print(f"\nTopic: {topic_id} ({len(posts)} English posts)")
        translations: dict[str, list[str]] = {"en": posts}
        for lang in ("es", "ar", "zh-CN"):
            key = "zh" if lang == "zh-CN" else lang
            print(f" Translating to {key}...")
            translations[key] = translate_batch(posts, lang)

        for idx in range(50):
            post_id = f"{topic_id}_{idx + 1:03d}"
            for lang in ("en", "es", "ar", "zh"):
                records.append(
                    {
                        "post_id": post_id,
                        "topic": topic_id,
                        "topic_label": TOPICS[topic_id]["label"],
                        "language": lang,
                        "language_name": LANGUAGES[lang],
                        "text": translations[lang][idx],
                    }
                )
    return records


def write_outputs(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Flat JSONL — one row per language version (ideal for embedding pipelines)
    jsonl_path = DATA_DIR / "posts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Nested JSON — one object per meaning, all languages together
    by_post: dict[str, dict] = {}
    for row in records:
        pid = row["post_id"]
        if pid not in by_post:
            by_post[pid] = {
                "post_id": pid,
                "topic": row["topic"],
                "topic_label": row["topic_label"],
                "texts": {},
            }
        by_post[pid]["texts"][row["language"]] = row["text"]

    nested = {
        "description": (
            "Toy multilingual social-media posts for testing whether embedding models "
            "cluster by semantic topic rather than by language under k-means."
        ),
        "topics": TOPICS,
        "languages": LANGUAGES,
        "n_topics": 3,
        "n_posts_per_topic": 50,
        "n_languages": 4,
        "n_records_flat": len(records),
        "posts": list(by_post.values()),
    }
    json_path = DATA_DIR / "posts.json"
    json_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # CSV for quick spreadsheet / pandas use
    csv_path = DATA_DIR / "posts.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("post_id,topic,topic_label,language,language_name,text\n")
        for row in records:
            text = row["text"].replace('"', '""')
            f.write(
                f'{row["post_id"]},{row["topic"]},"{row["topic_label"]}",'
                f'{row["language"]},{row["language_name"]},"{text}"\n'
            )

    meta = {
        "files": {
            "posts.json": "Nested posts with en/es/ar/zh texts per post_id",
            "posts.jsonl": "Flat records (one line per language version)",
            "posts.csv": "Same flat schema as JSONL",
        },
        "topics": list(TOPICS.keys()),
        "languages": list(LANGUAGES.keys()),
        "counts": {
            "posts_per_topic": 50,
            "topics": 3,
            "languages": 4,
            "unique_meanings": 150,
            "total_text_records": 600,
        },
        "clustering_notes": (
            "Ground-truth cluster label for semantic clustering is `topic`. "
            "Each `post_id` appears in 4 languages with the same meaning. "
            "A good multilingual embedder + k-means (k=3) should group records by topic, "
            "not by language."
        ),
    }
    (DATA_DIR / "README.md").write_text(
        "# Multilingual toy posts\n\n"
        "Artificial social-media posts for studying multilingual embeddings and k-means.\n\n"
        "## Topics\n\n"
        + "\n".join(f"- `{k}`: {v['label']}" for k, v in TOPICS.items())
        + "\n\n## Languages\n\n"
        + "\n".join(f"- `{k}`: {v}" for k, v in LANGUAGES.items())
        + "\n\n## Scale\n\n"
        "- 3 topics × 50 posts × 4 languages = **600** text records\n"
        "- 150 unique meanings (`post_id`), each with parallel translations\n\n"
        "## Files\n\n"
        "| File | Format |\n| --- | --- |\n"
        "| `posts.json` | Nested: each post has `texts.{en,es,ar,zh}` |\n"
        "| `posts.jsonl` | Flat: one JSON object per language version |\n"
        "| `posts.csv` | Flat CSV with the same columns |\n\n"
        "## Intended use\n\n"
        "Embed all 600 texts, run k-means with k=3, and compare clusters to `topic` "
        "(desired) versus `language` (undesired language silos).\n\n"
        f"```json\n{json.dumps(meta, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {DATA_DIR / 'README.md'}")


def main() -> None:
    for topic, posts in ENGLISH_POSTS.items():
        if len(posts) != 50:
            raise SystemExit(f"{topic}: expected 50 posts, got {len(posts)}")
    records = build_records()
    write_outputs(records)
    print(f"Done. {len(records)} records.")


if __name__ == "__main__":
    main()
