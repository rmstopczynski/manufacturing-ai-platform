# Manufacturing AI Intelligence Platform

An AI assistant for manufacturing engineers and plant managers that combines a **predictive
maintenance ML model** (structured sensor data) with a **RAG pipeline** (unstructured technical
documentation) and an **LLM orchestration layer**, so a user can ask a plain-language question
and get back an answer grounded in both real-time machine health and relevant documentation.

This project is motivated by problems I evaluated first-hand as a Data Governance Engineer Intern
at Jabil — specifically Informatica's CLAIRE AI for metadata classification, and an in-house AI
chatbot built for natural-language metadata search. This project builds an analogous system end
to end, from raw data through a deployed API and chat UI, so I own every layer of the stack rather
than evaluating tooling built by someone else.

**Status:** 🚧 In progress — see [Build Log](#build-log) below for what's done so far.

---

## What it does

- **Predicts machine failures** before they happen, using sensor/operational data (temperature,
  vibration, runtime hours) and a gradient-boosted classifier.
- **Answers engineering questions** by semantically searching internal technical documentation
  (manuals, maintenance logs, SOPs) instead of relying on keyword search.
- **Combines both** — when a user asks about a specific machine, the assistant pulls the ML
  prediction *and* relevant document context, then uses an LLM to explain the situation in plain
  language rather than returning a raw probability.

## Architecture

```
STRUCTURED DATA (sensor/machine logs)          UNSTRUCTURED DATA (manuals, SOPs, logs)
        │                                                 │
        ▼                                                 ▼
Pandas cleaning + feature engineering          Chunking + HF sentence embeddings
        │                                                 │
        ▼                                                 ▼
XGBoost predictive maintenance model           ChromaDB vector store
        │                                                 │
        └───────────────────┬─────────────────────────────┘
                             ▼
                LangChain orchestration layer
        (routes question → ML prediction / RAG retrieval / both →
         builds prompt → calls OpenAI API for the final answer)
                             │
                             ▼
                     FastAPI backend
                  (/predict, /ask endpoints)
                             │
                             ▼
                  Streamlit chat + dashboard
```

Packaged with Docker so the FastAPI service runs end to end from a fresh clone.

## Tech stack and why

| Tool | Role | Why |
|---|---|---|
| Pandas | Feature engineering on sensor data | Cleans/reshapes structured operational data before modeling |
| Scikit-learn / XGBoost | Predictive maintenance classifier | Gradient boosting handles tabular sensor data well and is explainable |
| Hugging Face sentence-transformers | Document embeddings | Standard, well-understood semantic search approach |
| ChromaDB | Vector store | Simple, well-documented retrieval API |
| LangChain | Orchestrates RAG + ML + LLM | Ties retrieval, prediction, and generation into one coherent flow |
| OpenAI API | Natural-language reasoning | Explains predictions/doc context in plain language |
| FastAPI | Serves the assistant as an API | Routes/request models/DI — conceptually close to ASP.NET Core |
| Streamlit | Chat UI + dashboard | Fast to build, keeps focus on the ML/RAG/LLM core |
| Docker | Packaging | Reproducible from a clone |

All data used (sensor logs, manuals, SOPs) is **synthetic** — no real Jabil data.

## Repo structure

```
manufacturing-ai-platform/
├── data/           # synthetic sensor/machine log data (raw + processed)
├── ml/             # feature engineering, model training, saved model artifacts
├── rag/            # synthetic documents, chunking/embedding, chroma_db
├── api/            # FastAPI app (/predict, /ask)
├── app/            # Streamlit chat + dashboard
├── docker/         # Dockerfile(s), docker-compose
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone <your-repo-url>
cd manufacturing-ai-platform
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI API key
```

---

## Build Log

### Phase 0 — Project setup ✅
- Repo structure created: `/data`, `/ml`, `/rag`, `/api`, `/app`, `/docker`.
- `requirements.txt` pinned for reproducibility.
- `.env.example` for API key configuration (real `.env` is gitignored).
- **Design decision:** kept `ml/` and `rag/` as separate top-level modules rather than one
  generic `pipeline/` folder, since they have genuinely different lifecycles (the model is
  retrained periodically; the vector store is rebuilt whenever documents change) and I want the
  README/code structure to make that distinction obvious to a reader.

### Phase 1 — Structured data + predictive model
*Not started yet.*

### Phase 2 — Unstructured data + RAG
*Not started yet.*

### Phase 3 — LangChain orchestration
*Not started yet.*

### Phase 4 — API + front end
*Not started yet.*

### Phase 5 — Evaluation harness
*Not started yet.*

### Phase 6 — Containerize + document
*Not started yet.*

---

## Defense-ready questions (for interviews)

- Walk me through what happens, step by step, when a user asks a question.
- Why LangChain instead of calling the OpenAI API directly?
- How did you decide on your chunking strategy, and what would you change at larger scale?
- Why XGBoost over a simpler model — what's the tradeoff?
- How do you evaluate whether the RAG system is actually working well?
- What happens if the retrieved documents are irrelevant to the question?
- How would this differ on real, sensitive manufacturing data instead of synthetic data?
- What would break first at thousands of machines and documents?

*(Answers to these get filled in as each phase is built, based on the actual decisions made —
not written in advance.)*
