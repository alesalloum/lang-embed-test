# AI datacenter claim–stance posts

Synthetic social-media posts for studying whether embeddings capture **stance**
(supportive / neutral / critical) on a shared topic: **Increasing number of AI datacenters**.

## Topic

- `ai_datacenters`: Increasing number of AI datacenters

## Aspects

- `economic`: Economic — Jobs, investment, taxes, energy prices, regional growth, capital markets.
- `environmental`: Environmental — Energy use, water, emissions, land use, renewables, climate tradeoffs.
- `infrastructure`: Infrastructure — Power grids, transmission, cooling, fiber, roads, construction capacity.
- `geopolitical`: Geopolitical — National security, chip supply, sovereignty, alliances, export controls.
- `local_community`: Local community — Zoning, noise, housing, schools, community benefits, local politics.
- `technological`: Technological — GPU clusters, cooling tech, efficiency, networking, model training scale.

## Stances

- `supportive`: Agrees with or frames the claim positively; expresses enthusiasm, confidence, or optimism.
- `critical`: Disagrees with, refutes, or frames the claim negatively; expresses concern, skepticism, or opposition.
- `neutral`: Expresses uncertainty, presents balanced/vague commentary, asks a question, or notes the situation without taking a side.

## Scale

- **500** claims × **3** stances = **1500** text records
- English only (`language=en`)
- Aspect mix: economic=84, environmental=84, infrastructure=83, geopolitical=83, local_community=83, technological=83

## Files

| File | Format |
| --- | --- |
| `claims.json` | Nested: each claim has `texts.{supportive,critical,neutral}` |
| `claims.jsonl` | Flat: one JSON object per stance post |
| `claims.csv` | Flat CSV with the same columns |
| `claim_catalog.jsonl` | One object per claim (claim text only) |

## How it was built

```bash
python3 scripts/generate_ai_datacenter_claims.py
```

Claims combine hand-authored stems with combinatorial frames across places,
actors, facilities, and aspect-specific effects. For each claim, three distinct
social-media-style posts are generated (supportive / critical / neutral).

## Flat schema (key fields)

- `claim_id` — shared across the three stance posts
- `post_id` — `{claim_id}_{stance}`
- `aspect` / `aspect_label` — topical facet within AI datacenters
- `claim` — underlying proposition
- `stance` — `supportive` | `critical` | `neutral`
- `text` — social-media post expressing that stance toward the claim

## Intended use

Embed all 1500 texts and evaluate whether vectors separate by `stance`
while still reflecting shared `claim_id` / `aspect` semantics — exploratory
tests for stance-aware embedding quality.

```json
{
  "files": {
    "claims.json": "Nested: each claim has texts.{supportive,critical,neutral}",
    "claims.jsonl": "Flat records (one line per stance post)",
    "claims.csv": "Same flat schema as JSONL",
    "claim_catalog.jsonl": "One line per claim (no stance texts)"
  },
  "topic": "ai_datacenters",
  "stances": [
    "supportive",
    "critical",
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
  "language": "en",
  "counts": {
    "claims": 500,
    "stances_per_claim": 3,
    "total_text_records": 1500,
    "aspect_counts": {
      "economic": 84,
      "environmental": 84,
      "infrastructure": 83,
      "geopolitical": 83,
      "local_community": 83,
      "technological": 83
    }
  },
  "embedding_notes": "Ground-truth label for stance separation is `stance`. Each `claim_id` appears in 3 stance variants with the same underlying claim. A good embedding should separate supportive / neutral / critical while keeping same-claim posts nearer in topic space than unrelated claims."
}
```
