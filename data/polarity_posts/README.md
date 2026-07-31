# AI datacenter phenomenon-polarity posts

Synthetic social-media posts labeled by **polarity toward AI
datacenter expansion** (not claim-agreement stance).

## Topic

- `ai_datacenters`: Increasing number of AI datacenters

## Polarities

- `pro`: Favors AI datacenter expansion; frames buildout as beneficial, necessary, or worth supporting.
- `against`: Opposes AI datacenter expansion; frames buildout as harmful, unnecessary, or worth resisting.
- `neutral`: Does not take a side on whether AI datacenters should expand; hedges, asks questions, or presents tradeoffs without a verdict.

## Aspects

- `economic`: Economic — Jobs, investment, taxes, energy prices, regional growth, capital markets.
- `environmental`: Environmental — Energy use, water, emissions, land use, renewables, climate tradeoffs.
- `infrastructure`: Infrastructure — Power grids, transmission, cooling, fiber, roads, construction capacity.
- `geopolitical`: Geopolitical — National security, chip supply, sovereignty, alliances, export controls.
- `local_community`: Local community — Zoning, noise, housing, schools, community benefits, local politics.
- `technological`: Technological — GPU clusters, cooling tech, efficiency, networking, model training scale.

## Scale

- **300** posts × **3** polarities = **900** texts
- English only (`language=en`)
- Aspect mix: economic=150, environmental=150, infrastructure=150, geopolitical=150, local_community=150, technological=150

## Files

| File | Format |
| --- | --- |
| `posts.json` | Array of posts |
| `posts.jsonl` | One object per post |
| `posts.csv` | Flat CSV |
| `info.json` | Counts + labeling notes |

## How it was built

```bash
python3 scripts/generate_ai_datacenter_polarity.py
```

## Schema (key fields)

- `post_id` — unique id
- `polarity` — `pro` | `against` | `neutral`
- `aspect` / `aspect_label` — topical facet
- `text` — social-media post expressing that polarity

## Intended use

Embed all posts and test whether vectors separate by `polarity`
toward AI datacenter expansion (vanilla vs polarity-instruct).

```json
{
  "topic": "ai_datacenters",
  "topic_label": "Increasing number of AI datacenters",
  "language": "en",
  "polarities": [
    "pro",
    "against",
    "neutral"
  ],
  "aspects": [
    "economic",
    "environmental",
    "infrastructure",
    "geopolitical",
    "local_community",
    "technological"
  ],
  "counts": {
    "total": 900,
    "per_polarity": {
      "pro": 300,
      "against": 300,
      "neutral": 300
    },
    "aspect_counts": {
      "economic": 150,
      "environmental": 150,
      "infrastructure": 150,
      "geopolitical": 150,
      "local_community": 150,
      "technological": 150
    },
    "polarity_aspect": {
      "pro": {
        "economic": 50,
        "environmental": 50,
        "infrastructure": 50,
        "geopolitical": 50,
        "local_community": 50,
        "technological": 50
      },
      "against": {
        "economic": 50,
        "environmental": 50,
        "infrastructure": 50,
        "geopolitical": 50,
        "local_community": 50,
        "technological": 50
      },
      "neutral": {
        "economic": 50,
        "environmental": 50,
        "infrastructure": 50,
        "geopolitical": 50,
        "local_community": 50,
        "technological": 50
      }
    }
  },
  "files": {
    "posts.json": "Array of polarity posts",
    "posts.jsonl": "One JSON object per post",
    "posts.csv": "Flat CSV with the same columns"
  },
  "labeling_notes": "Ground-truth label is `polarity` toward the phenomenon of AI datacenter expansion (pro / against / neutral), NOT agreement with a specific claim. Aspect is a topical facet used for secondary encoding in plots.",
  "seed": 42,
  "n_per_polarity": 300
}
```
