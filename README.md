# Fluent UDF Copilot

Retrieval-grounded assistant that helps engineers write, explain, and debug ANSYS Fluent User-Defined Functions (UDFs). The system indexes the Fluent Customization Manual plus a small curated library of example UDFs, retrieves relevant passages with FAISS (plus optional BM25 hybrid), and prompts a small instruction-tuned LLM with that grounded context. A lightweight static checker validates the output structurally before it is returned.

---

## Project layout

```
.
├── Ansys_Fluent_UDF_Manual.pdf        # Primary knowledge source (provided)
├── requirements.txt
├── .env.example                       # Environment template (copy to .env)
├── data/
│   ├── eval_prompts.json              # Curated baseline vs RAG prompt set
│   └── examples/                      # Seed example UDFs indexed alongside the manual
├── src/
│   ├── ingest.py                      # PDF parsing + chunking + metadata
│   ├── retriever.py                   # Sentence-Transformers + FAISS (+ optional BM25)
│   ├── llm_client.py                  # OpenAI-compatible chat client
│   ├── prompts.py                     # Baseline + RAG prompt templates
│   ├── checker.py                     # Static UDF structural validator
│   ├── assistant.py                   # Ties retrieval, prompting, LLM, checker together
│   ├── evaluate.py                    # Evaluation harness
│   └── config.py                      # Env-backed configuration
├── scripts/
│   ├── build_index.py                 # One-time index build
│   ├── query.py                       # Single-question CLI
│   └── evaluate.py                    # Baseline vs RAG comparison runner
├── index/                             # Created after first index build (FAISS + chunks.jsonl)
└── artifacts/                         # Evaluation records land here
```

---

## 1. Install

Requires **Python 3.10+** on Windows/macOS/Linux.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, you can allow scripts only for the current PowerShell session and then activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux activation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first run of retrieval will download the embedding model
(`sentence-transformers/all-MiniLM-L6-v2`, ~90 MB) into your Hugging Face cache.
If CUDA is available, embeddings can run on GPU (`EMBED_DEVICE=auto` by default).

## 2. Configure the LLM backend

Copy `.env.example` to `.env` and edit values as needed.

Quick way to create `.env` from PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

### Option A: Local Ollama (recommended for offline work)

```powershell
# Install Ollama from https://ollama.com and then:
ollama pull llama3.2:3b
ollama serve  # usually starts automatically on install
```

Core `.env` values for local Ollama:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2:3b
```

### Option B: OpenAI / other cloud provider

```
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

Any endpoint that speaks the OpenAI Chat Completions schema works
(LM Studio, vLLM, Together AI, Groq, etc.).

### Embedding GPU/CPU settings

Add these in `.env` if you want explicit control:

```
EMBED_DEVICE=auto
EMBED_BATCH_SIZE=128
```

- `EMBED_DEVICE=auto` selects `cuda` when available, else `cpu`.
- `EMBED_DEVICE=cuda` forces GPU and falls back to CPU if CUDA is unavailable.
- Increase `EMBED_BATCH_SIZE` on larger GPUs for faster indexing/query embedding.

## 3. Build the retrieval index (one-time)

```powershell
python -m scripts.build_index
```

This reads `Ansys_Fluent_UDF_Manual.pdf` plus everything in `data/examples/`, chunks the
content (~200 tokens with 80-token overlap by default), embeds it, and writes a FAISS
index to `index/`. Expect a few minutes on first run because the full manual is 852
pages. Use `--limit-pages 60` for a quick smoke build while you iterate.

If the manual file is not tracked in your remote repository, place
`Ansys_Fluent_UDF_Manual.pdf` in the project root before running `build_index`.

Useful flags:

```powershell
python -m scripts.build_index --limit-pages 60           # fast smoke build
python -m scripts.build_index --chunk-tokens 500         # larger chunks
python -m scripts.build_index --embed-model BAAI/bge-small-en-v1.5
python -m scripts.build_index --embed-device cuda --embed-batch-size 128
```

## 4. Ask the copilot a question

```powershell
python -m scripts.query "write a parabolic inlet velocity UDF for a 2D channel"
python -m scripts.query --mode baseline "explain when to use DEFINE_SOURCE"
python -m scripts.query --retrieve-only "DEFINE_PROFILE F_CENTROID example"
```

Each answer comes with (a) the retrieved context tags, (b) the model answer in the
required structure, and (c) the static-check report (missing `#include`, bad signature,
hallucinated tokens, etc.).

## UI app (chat + reports)

Run the Streamlit UI:

```powershell
streamlit run app.py
```

What you get in the UI:

- Chat interface for generation/explanation/debug prompts.
- Model/backend selection in the sidebar (Ollama/OpenAI-compatible).
- **RAG panel** on the Chat tab (OFF = baseline, ON = retrieval-grounded): includes **Top-K** plus the RAG toggle in one bordered panel.
- Separate report tabs:
  - Retrieval Report (top chunks, scores, source sections)
  - Checker Report (static pass/fail + warnings)
  - Evaluation Report (reads `artifacts/eval_records.json` and summarizes metrics)

## 5. Run the evaluation

```powershell
python -m scripts.evaluate
```

This runs every prompt in `data/eval_prompts.json` through both `baseline` and `rag`
modes, saves the full records to `artifacts/eval_records.json`, and prints a summary:

| mode     | prompts | macro_accuracy | structural_pass_rate | citation_rate |
|----------|---------|----------------|----------------------|---------------|
| baseline |   14    |     xx.x%      |         xx.x%        |     n/a       |
| rag      |   14    |     xx.x%      |         xx.x%        |     xx.x%     |

You can also run a single mode:

```powershell
python -m scripts.evaluate --modes rag
```

---

## Metrics

1. **Macro-selection accuracy** - fraction of code answers whose `DEFINE_*` matches the
   expected family in `eval_prompts.json`.
2. **Structural pass rate** - static checks: presence of `#include "udf.h"`, a valid
   `DEFINE_*` signature, and no unknown `DEFINE_*` / `C_*` tokens (the allow-list is
   harvested from the indexed corpus).
3. **Citation rate (RAG)** - whether the answer cites any retrieved context tag like
   `[C1]`.
4. **Usefulness (1-5)** - scored by hand against the saved `artifacts/eval_records.json`.

---

## How the pieces fit together

```
 ┌───────────────────┐    ┌─────────────────────┐     ┌────────────────────┐
 │  Manual PDF +     │    │  ingest.py          │     │  FAISS index       │
 │  example UDFs     │──▶ │  (chunk+metadata)   │──▶  │  + chunks.jsonl    │
 └───────────────────┘    └─────────────────────┘     └─────────┬──────────┘
                                                                 │
 ┌───────────────────┐    ┌─────────────────────┐     ┌────────▼──────────┐
 │  User question    │──▶ │  retriever.py       │──▶  │  Top-k passages   │
 └───────────────────┘    │  (embed + hybrid)   │     └────────┬──────────┘
                          └─────────────────────┘              │
                                                               ▼
                          ┌─────────────────────┐     ┌────────────────────┐
                          │  prompts.py         │──▶  │  LLM (OpenAI/      │
                          │  (system + context) │     │  Ollama compat.)   │
                          └─────────────────────┘     └────────┬──────────┘
                                                               │
                          ┌─────────────────────┐              ▼
                          │  checker.py         │   ┌────────────────────┐
                          │  (static validator) │◀──│  Model answer      │
                          └─────────────────────┘   └────────────────────┘
```

---
