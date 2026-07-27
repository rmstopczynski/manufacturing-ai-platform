"""
SANDBOX-ONLY TEST — not part of the shipped pipeline.

This sandbox can't reach api.openai.com, so this substitutes LangChain's
FakeListChatModel for ChatOpenAI purely to verify that routing, prediction-fetching,
retrieval, and prompt construction all wire together correctly end to end, and that
the LCEL chain (prompt | llm | parser) actually executes. The fake model just echoes
back a canned response — the point is to confirm the REAL PROMPT SENT TO THE MODEL
looks correct, and that edge cases (unknown machine, no relevant docs, vague question)
are handled sanely before the LLM is ever called.

Run `python3 -m orchestration.chain` (the real entry point) locally with a real
OPENAI_API_KEY to get an actual LLM-generated answer.
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import json
from sklearn.feature_extraction.text import TfidfVectorizer

import orchestration.chain as chain_module
from orchestration.chain import ManufacturingAssistant, build_chain, SYSTEM_PROMPT


def _build_offline_retrieve_stub():
    """TF-IDF stand-in for rag.retrieve.retrieve — same reasoning as
    rag/_sandbox_test_retrieval.py: this sandbox can't download the real embedding
    model, so this substitutes a local TF-IDF search over the same document manifest
    just to exercise the chain's control flow and prompt construction."""
    with open("rag/documents/manifest.json") as f:
        docs = json.load(f)
    texts = [f"{d['title']}\n\n{d['body']}" for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    def stub_retrieve(query: str, k: int = 3, machine_type: str = None):
        candidates = list(range(len(docs)))
        if machine_type:
            candidates = [i for i in candidates
                          if docs[i]["machine_type"] in (machine_type, "GENERAL")]
        if not candidates:
            return []
        q_vec = vectorizer.transform([query])
        sims = (matrix[candidates] @ q_vec.T).toarray().flatten()
        ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])[:k]
        return [{
            "text": f"{docs[i]['title']}\n\n{docs[i]['body']}",
            "doc_type": docs[i]["doc_type"],
            "machine_type": docs[i]["machine_type"],
            "title": docs[i]["title"],
            "distance": round(1 - sim, 3),  # convert similarity -> distance-like scale
        } for i, sim in ranked]

    return stub_retrieve

TEST_CASES = [
    "Is machine M001 at risk of failing soon?",       # known machine -> prediction + retrieval
    "Is machine M999 at risk of failing soon?",        # unknown machine -> graceful error
    "What's the lockout/tagout procedure?",            # no machine ID -> retrieval only
    "asdf",                                            # too vague -> short-circuit, no LLM call
    "Tell me about quantum computing trends",          # no relevant docs -> should say so
]


def run():
    chain_module.retrieve = _build_offline_retrieve_stub()  # monkeypatch, sandbox-only

    fake_llm = FakeListChatModel(responses=["[FAKE LLM RESPONSE — see prompt above]"] * 10)
    assistant = ManufacturingAssistant(llm=fake_llm)

    for q in TEST_CASES:
        print("=" * 80)
        print(f"QUESTION: {q}")
        result = assistant.answer(q)
        print(f"ROUTING: {result.get('routing')}")
        pred = result.get("prediction")
        if pred:
            print(f"PREDICTION: found={pred['found']}, "
                  f"{'prob=' + str(pred.get('failure_probability')) if pred['found'] else pred.get('error')}")
        chunks = result.get("retrieved_chunks", [])
        print(f"RETRIEVED CHUNKS: {len(chunks)} "
              f"{[c['title'] for c in chunks] if chunks else ''}")
        print(f"ANSWER: {result['answer']}")
        print()


if __name__ == "__main__":
    run()
