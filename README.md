# Forward Checker

Paste a WhatsApp forward (Hindi, English or Hinglish) and get:
1. the closest fact-check already published by an Indian fact-checker
2. a verdict based on that fact-check, or **Unverified** if nothing matches
3. a polite reply you can post in the family group

> Work in progress. See [PLAN.md](PLAN.md) for the full build plan.

## Status

Phase 1 (GenAI core):

| Step | What | Status |
|---|---|---|
| Data | 16,953 Indian fact-checks (English + Hindi) cleaned into SQLite (`notebooks/01_load_factchecks.ipynb`) | ✅ |
| Embeddings + Chroma | Qwen3 embeddings (local, chosen from a 4-model comparison in `notebooks/02_embeddings_chroma.ipynb`), `create_database.py` builds the Chroma DB, threshold test in `notebooks/03_search_threshold.ipynb` | ✅ |
| Verdict chain | `main.py`: Groq LLM checks "same claim?", Pydantic verdict + Hindi/English reply, Unverified below threshold | ✅ |
| Fresh data | On demand: when the local DB has no match, ask the live Google Fact Check API and save new fact-checks back into the DB | ✅ |
| App + eval + deploy | Streamlit, 30-forward evaluation, Hugging Face Spaces | — |

Later phases (GenAI extras, data + ML add-on, agent) are in [PLAN.md](PLAN.md).

## Setup

Needs Python 3.12 (TensorFlow and Chroma lag behind newer Python versions).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name forward-checker --display-name "forward-checker (3.12)"
cp .env.example .env   # then fill in your own keys, never commit .env
```

Then run `notebooks/01_load_factchecks.ipynb` with the **forward-checker (3.12)** kernel. It downloads the fact-check feed and builds `data/factchecks.db` (the `data/` folder is not on GitHub, it's rebuilt by the notebook).

## Data

Fact-checks come from the [Data Commons ClaimReview feed](https://datacommons.org/factcheck/download) (Google, CC BY), filtered to Indian publishers. Only the claim, rating, publisher, date and link are stored, never the full article. Every result links back to the original fact-check.

## Prior work

I looked at these projects before building mine:
- [Timhongphuc/Fakify](https://github.com/Timhongphuc/Fakify) (MIT): fact-checks an article URL with live web search and an LLM. Gave me the starting idea. No code reused.
- [md-emon-hasan/informatruth](https://github.com/md-emon-hasan/informatruth) (MIT): fake-news classifier with prompt-injection guards and a majority-class baseline in its results. Ideas only.

What's different here: WhatsApp forward text instead of URLs, Indian fact-checkers in Hindi and English, an "Unverified" answer when there's no evidence, and a ready-to-send reply.

## License

MIT, see [LICENSE](LICENSE).
