# lang-embed-test

Toy data and experiments for multilingual embedding models: do same-meaning posts cluster by **topic**, or do they split by **language** under k-means?

## Data

See [`data/`](data/README.md):

- 3 topics × 50 posts × 4 languages (en, es, ar, zh) = **600** texts
- Topics: AI coding innovations, AI copyright theft, AI mass surveillance
- Parallel translations share a `post_id`

Regenerate with:

```bash
pip install deep-translator
python scripts/generate_toy_posts.py
```
