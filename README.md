# AI CV Screener (Streamlit + LlamaIndex) — with synthetic CV generator

A small end-to-end demo that **creates synthetic tech CVs**, builds a **RAG index** over them, and exposes a **Streamlit chat app** to query candidate profiles using natural language.

---

## ✨ What’s inside

* **Synthetic CV generator**: async pipeline that produces multiple PDF CVs with realistic content and an optional AI-generated avatar.
* **RAG pipeline**: indexes the generated PDFs with LlamaIndex and answers questions grounded in the CVs.
* **Streamlit UI**: clean chat interface to query the CVs and inspect retrieved sources.

---

## 🗂 Project layout (suggested)

> The code expects a `utils/` package for internal modules. If you don’t use this layout, update imports accordingly.

```
.
├─ app.py                  # Streamlit chat app
├─ generate_cvs.py         # Orchestrates async generation of many CV PDFs
├─ utils/
│  ├─ cv.py                # Single-CV generation (identity + LLM + PDF + avatar)
│  ├─ rag.py               # RAG pipeline (indexing + queries over PDFs)
│  └─ avatars.py           # Avatar generation via OpenRouter
├─ data/
│  └─ vector_store/        # (auto) LlamaIndex persisted index
├─ cvs/                    # (auto) Generated CV PDFs land here
├─ avatars/                # (auto) Generated avatar images
├─ .env                    # Your environment variables
└─ pyproject.toml          # uv project dependencies
```

If you keep the files at repo root (as provided), create the `utils/` package and move them:

```bash
mkdir -p utils
git mv cv.py utils/cv.py
git mv rag.py utils/rag.py
git mv avatars.py utils/avatars.py
```

---

## 🧰 Requirements

* Python **3.12+**
* [uv](https://docs.astral.sh/uv/) (fast Python package manager)
* OpenAI API key (for LLM + embeddings) — `OPENAI_API_KEY`
* OpenRouter API key (for avatar generation) — `OPEN_ROUTER_KEY` *(optional if you skip avatars)*

---

## ⚙️ Setup with uv

```bash
# 1) Ensure Python 3.12 is available to uv
uv python install 3.12

# 2) Create a virtual environment and install deps
uv sync

# 3) Activate the environment (if you prefer)
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell
```

### Environment variables

Create a `.env` file in the project root:

```env
# --- OpenAI ---
OPENAI_API_KEY=sk-...

# LLM / Embedding models (defaults shown)
OPENAI_LLM_MODEL=gpt-4o
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_MODEL=gpt-5-nano         # used by the CV agent (can be any chat model you have)

# --- Avatar generation via OpenRouter (optional) ---
OPEN_ROUTER_KEY=or-...

# --- Generation controls ---
CVS_OUTPUT_DIR=cvs
MAX_CONCURRENT=5                # parallel tasks for bulk generation
CV_TO_GENERATE=30               # how many CVs to produce in bulk
```

> The app uses `python-dotenv`, so `.env` is loaded automatically.

Install any missing system fonts if your PDFs need specific typography; ReportLab will fall back to built-ins otherwise.

---

## 🧪 Generate a dataset of CVs

First, produce some synthetic CV PDFs (plus avatars) so the RAG index has content:

```bash
uv run python generate_cvs.py
```

* PDFs are written to `./cvs/`.
* Avatars (PNG/JPG) are written to `./avatars/`.
* Concurrency and output counts are controlled by `MAX_CONCURRENT` and `CV_TO_GENERATE` in `.env`.

> If you don’t want avatars, either remove/disable that step in `utils/cv.py` or skip setting `OPEN_ROUTER_KEY` (avatar generation will raise if called without a key).

---

## ▶️ Run the Streamlit app

```bash
uv run streamlit run app.py
```

What it does:

1. Loads/creates the **LlamaIndex** vector store from `./cvs` (PDFs only).
2. Caches the RAG pipeline in the Streamlit session.
3. Provides a **chat input** to ask questions like:

   * “Who has production experience with FastAPI?”
   * “Find candidates with ML Ops and Kubernetes.”

The app shows answers plus collapsible **source previews** so you can see which PDF chunks were used.

---

## 🧠 RAG internals

* **Indexing**: PDFs in `./cvs` are read and chunked (configurable size/overlap) and embedded with OpenAI embeddings.
* **Persistence**: the vector index is persisted under `./data/vector_store/`.
* **Querying**: similarity search returns top-K chunks; a compact LLM response is generated using LlamaIndex’s query engine.

You can also provide a **custom system prompt** at query time (see `custom_query` in `utils/rag.py`).

---

## 🔧 Configuration knobs

* **Models**

  * App LLM: `OPENAI_LLM_MODEL` (default: `gpt-4o`)
  * Embeddings: `OPENAI_EMBED_MODEL` (default: `text-embedding-3-small`)
  * CV agent model: `OPENAI_MODEL` (defaults to `gpt-5-nano` placeholder — use whatever you have access to)

* **Chunking / Retrieval**

  * `chunk_size`, `chunk_overlap`, and `top_k` can be tuned in the `CVScreenerRAG` initializer.

* **Paths**

  * `cvs/` for PDFs
  * `data/vector_store/` for the index
  * `avatars/` for headshots

---

## 🐛 Troubleshooting

* **Module import errors (`utils.*`)**
  Ensure you created the `utils/` package (see layout above) or update imports in `app.py` and `generate_cvs.py` to point to your actual file locations.

* **Avatar generation fails**

  * Verify `OPEN_ROUTER_KEY` is set and valid.
  * The OpenRouter image model (e.g., `google/gemini-2.5-flash-image`) must be enabled for your key.
  * Network egress must be allowed to `openrouter.ai`.

* **Index not picked up / empty answers**

  * Make sure `./cvs` contains PDFs before you start the app.
  * Delete `./data/vector_store/` to force a re-index after adding new PDFs.

* **Streamlit hot-reload and caching**
  The RAG pipeline is cached as a resource. If you change RAG code/params, stop/restart the app.

---

## 📈 Roadmap (ideas)

* Per-file metadata panels and full-text previews.
* Feedback loop (thumbs up/down) to refine prompts/chunking.
* Multi-vector retrieval (skills/titles) and reranking.
* Export shortlists to CSV/ATS.

