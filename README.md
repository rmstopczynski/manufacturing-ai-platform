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

### Phase 1 — Structured data + predictive model ✅

**Data generation (`data/generate_sensor_data.py`):** Simulated 60 machines across 5 types
(CNC mill, conveyor, press, robot arm, pump) over 180 days of daily readings. ~35% of machines
are assigned a failure event; those machines get a 30-day pre-failure "drift" — rising
temperature and vibration that ramps up to the failure day — while healthy machines fluctuate
around a stable per-type baseline. This produced 9,880 rows with a **1.06% positive rate**
(`failed_within_7d = 1`), which is deliberately imbalanced to mirror what real predictive
maintenance data looks like (failures are rare events, not 50/50).

**Feature engineering (`ml/feature_engineering.py`):** Point-in-time sensor readings are a weak
signal on their own — what actually indicates an approaching failure is a machine drifting away
from *its own* recent baseline. So each row gets: 7-day rolling mean/std of temperature and
vibration, and a 7-day rate-of-change (today vs. 7 days ago), on top of the raw readings,
`runtime_hours`, `days_since_maintenance`, and one-hot encoded `machine_type`.

**Modeling (`ml/train_model.py`):** Two models trained side by side on purpose — Logistic
Regression as an interpretable baseline, and XGBoost as the model the project is built around —
so "why XGBoost over something simpler" has a real number behind it instead of a talking point.

- **Split:** time-based (train on the first ~80% of calendar days, test on the last ~20%), not a
  random split. A random split would leak information — nearby days for the same machine share a
  rolling window, so random shuffling lets the model "peek" at data adjacent to its test rows.
- **Metric:** precision/recall/F1, not accuracy. With a 1% positive rate, a model that always
  predicts "healthy" scores ~99% accuracy while being useless.
- **Imbalance handling:** `class_weight="balanced"` for logistic regression;
  `scale_pos_weight = negatives/positives` for XGBoost — both compensate for the rare positive
  class without synthetically oversampling rows.

**Actual results on the held-out test set (41 real failure-labeled rows):**

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.532 | **1.000** | **0.695** |
| XGBoost | 0.530 | 0.854 | 0.654 |

**Honest design-decision note:** On this dataset, logistic regression actually edges out XGBoost
on F1 — it caught every single failure in the test set (perfect recall), at a similar precision
to XGBoost. This is a genuinely useful thing to be able to explain rather than a problem to hide:
with only 64 positive examples in the training set, a simpler, heavily-regularized linear model
generalizes at least as well as a more flexible gradient-boosted tree ensemble, which has more
capacity to overfit sparse minority-class signal. The tradeoff XGBoost usually offers — capturing
non-linear interactions and threshold effects in sensor readings — needs more positive examples
to pay off than this synthetic dataset happens to contain. XGBoost is still the one persisted for
the API (`ml/models/xgb_predictive_maintenance.joblib`), both because it's the one the rest of
the pipeline is built to showcase and because feature importances give a cleaner story for the
LLM layer to reason over — but the honest comparison is the more valuable interview answer than
a cherry-picked "XGBoost wins" result would have been.

**Feature importance (XGBoost):** `temp_roll7_std` (rolling temperature volatility) dominates at
37% importance — machines that are about to fail don't just run hotter, their temperature
becomes *less stable* day to day, which the rolling std captures directly. Machine type
(especially CNC mills and conveyors) and rolling vibration mean are the next largest contributors.
Interestingly, the raw point-in-time `temperature_c` and `vibration_mm_s` readings and the
rate-of-change features contribute very little — confirming the original hypothesis that trend
features (rolling stats) carry the real signal, not instantaneous values.

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
