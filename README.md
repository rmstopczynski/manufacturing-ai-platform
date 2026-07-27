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
├── ml/             # feature engineering, model training, prediction, saved model artifacts
├── rag/            # synthetic documents, chunking/embedding, chroma_db, retrieval
├── orchestration/  # routing + LangChain chain tying ML + RAG + LLM together
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

### Phase 2 — Unstructured data + RAG ✅

**Documents (`rag/generate_documents.py`):** 24 synthetic documents — 5 equipment manuals (one
per machine type), 7 SOPs (2 general: lockout/tagout and maintenance scheduling; 5 per-machine-type
troubleshooting SOPs), and 12 maintenance log entries. The maintenance logs are deliberately mixed:
some describe a real failure that matched the documented pre-failure pattern, some describe false
alarms or "no issue found" inspections, and one describes a sensor malfunction that mimicked a
real trend. This matters for RAG realism — a retrieval system that only ever sees confirmed-failure
logs will bias the LLM layer toward always concluding "this is a real failure," which isn't how
real maintenance data looks.

**Chunking strategy (`rag/build_index.py`):** At this corpus's scale, each document is short (a
manual section, an SOP, or a single log entry — typically 100-250 words), so the chunking unit is
**one document = one chunk** for anything under 1,200 characters, rather than splitting into
smaller fixed-size pieces. Splitting a short document further would risk cutting a single warning
sign or procedure step across chunk boundaries, which actively hurts retrieval rather than helping
it — there's no benefit to sub-document chunking when the whole document is already
retrieval-sized. A character-based overlapping splitter (150-char overlap) is included for
anything that exceeds the threshold, and it's genuinely exercised: 5 of the 24 documents (the
equipment manuals, which run slightly longer) came in just over 1,200 characters and were split
into 2 chunks each, giving 29 total chunks from 24 documents.

**What would change at larger scale:** this one-document-one-chunk approach does not scale to
real multi-page manuals. At that scale I'd move to either fixed-size overlapping chunks (e.g.
500 tokens with 50-token overlap) or section-aware splitting on the manual's own headers, so a
100-page manual doesn't get embedded as a single unsearchable blob or, at the other extreme,
shredded into fragments that lose surrounding context.

**Embeddings + vector store:** chunks are embedded with `all-MiniLM-L6-v2`
(sentence-transformers, 384-dim) — a small, fast, well-understood model that doesn't require
digging into transformer internals to reason about, appropriate for a 24-document corpus — and
loaded into a persistent ChromaDB collection (`rag/chroma_db/`) with metadata (`doc_type`,
`machine_type`, `title`) attached to every chunk so retrieval can be filtered (e.g. "only PUMP
docs plus GENERAL SOPs") rather than searching the whole corpus indiscriminately.

**Retrieval (`rag/retrieve.py`):** a `retrieve(query, k, machine_type)` function that the Phase 3
orchestration layer will call directly — it embeds the query, optionally filters to a specific
machine type (plus `GENERAL` docs, which apply across all types), and returns the top-k chunks
with their distance scores.

**Retrieval sanity check:** this sandbox environment's network can't reach huggingface.co to
download the actual embedding model, so I validated the chunking → storage → retrieval pipeline
end to end with a TF-IDF stand-in embedding (`rag/_sandbox_test_retrieval.py`, not part of the
shipped pipeline) instead. Results were sensible: a query about "vibration climbing on a CNC mill"
correctly surfaced the CNC manual, the CNC troubleshooting SOP, and the CNC maintenance log
(in that order); a "pump bearing" query surfaced pump-specific docs; a general maintenance-timing
query surfaced the GENERAL scheduling SOP. `build_index.py` and `retrieve.py` themselves are
unchanged and use the real `all-MiniLM-L6-v2` model — they'll run correctly the moment you run
them with normal internet access locally.

### Phase 3 — LangChain orchestration ✅

**Routing (`orchestration/router.py`):** given a raw question, decides (a) whether it needs an ML
prediction — only if a specific machine ID (regex `M\d{3}`) is mentioned, since predictions are
per-machine — and (b) what to filter retrieval to, based on machine-type keywords in the question
(e.g. "conveyor," "pump"). **Deliberately plain Python, not a second LLM call.** With only two
real branches (prediction needed or not; which machine type to filter to), asking an LLM "what
should I do with this question?" would add latency and cost for a decision a regex handles in
microseconds. This is a genuine scope-appropriate tradeoff — a more open-ended assistant with many
tools would justify LLM-based routing or a proper agent framework; this doesn't, and claiming
otherwise would be overselling the project.

**A real gap found and fixed during testing:** the first version only inferred machine *type* from
keywords in the question text (e.g. "conveyor"), so a question like "Is M001 at risk?" — which
names a specific machine but never says what kind of machine it is — fell back to searching the
*entire* document corpus instead of filtering to CNC-mill-specific docs. Fixed by having
`ml/predictor.py` resolve and return the machine's type (derived from the one-hot `type_*`
feature columns) alongside its prediction, so the orchestration layer can fall back to that when
the question itself gives no type keyword. Caught this by actually running test cases through the
pipeline, not by reasoning about the code in the abstract — a good example of why Phase 5
(evaluation harness) matters.

**ML prediction (`ml/predictor.py`):** loads the persisted XGBoost model and the most recent
feature row for the requested machine ID, returning a structured result (probability, days since
maintenance, top 3 contributing features by importance) or a clean "machine not found" error
rather than crashing — this is what lets the LLM say "I don't have data for that machine ID"
instead of guessing.

**Prompt construction (`orchestration/chain.py`):** a system prompt instructs the model to answer
*only* from the provided ML prediction + retrieved documents, not to speculate about a machine's
health without a prediction, and to say plainly when nothing relevant was retrieved rather than
falling back to general knowledge. Retrieved chunks above a distance threshold are filtered out
before ever reaching the prompt — instead of always injecting the top-k regardless of relevance,
which is what would let the model quietly answer from documents that don't actually address the
question. The LLM call itself is a small LCEL chain
(`ChatPromptTemplate | ChatOpenAI | StrOutputParser`) — LangChain's actual job here is templating
and swappability (decoupling the prompt/model from the call site so a different model or added
structured-output parsing later doesn't require touching the calling code), not magic; for a
single-model, single-prompt project like this, a raw OpenAI call would work almost as well, and
that's the honest answer if pushed on "why LangChain" in an interview rather than overselling it.

**Edge cases handled (and tested):**
- **Vague input** (e.g. a single nonsense word): short-circuits before the LLM is ever called,
  judged by word count rather than character count, since a short garbage string and a short real
  string aren't distinguishable by length alone.
- **Unknown machine ID:** `predict_machine` returns a structured "not found" result; the prompt
  tells the LLM this explicitly so it says so instead of inventing a plausible-sounding answer.
- **No relevant documents retrieved** (e.g. an off-topic question): the retrieval context passed
  to the LLM explicitly states nothing relevant was found, rather than passing the top-k anyway
  and hoping the model notices they're irrelevant.

**Testing note (sandbox limitation):** this sandbox can't reach `api.openai.com` or
`huggingface.co`, so end-to-end testing here used LangChain's `FakeListChatModel` in place of
`ChatOpenAI` (`orchestration/_sandbox_test_chain.py`, not part of the shipped pipeline) to verify
routing, prediction-fetching, retrieval, and prompt construction all wire together correctly — the
exact formatted prompt sent to the model was inspected directly to confirm it's correct, not just
that the code runs without errors. `orchestration/chain.py` and `ml/predictor.py` are otherwise
unchanged production code and will call the real OpenAI API correctly once run locally with
`OPENAI_API_KEY` set.

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
