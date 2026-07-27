"""
SANDBOX-ONLY TEST — not part of the shipped pipeline.

Runs eval/run_eval.py's harness logic against a fake LLM + offline TF-IDF retrieval
stub (same reasoning as orchestration/_sandbox_test_chain.py) to verify the eval
harness ITSELF is correct — that checks fire correctly, structural checks pass/fail
as expected, and the report format works — before trusting it to run for real
locally against Groq. Content checks will mostly fail here since the fake LLM
returns a canned string, not a real answer; that's expected and fine, since the
point is validating the harness mechanics, not getting a real pass rate.

Run `python -m eval.run_eval` (the real entry point) locally with a real API key
to get a real evaluation report.
"""

import json
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import orchestration.chain as chain_module
from orchestration.chain import ManufacturingAssistant
from eval.run_eval import run, print_report


def _offline_retrieve_stub():
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open("rag/documents/manifest.json") as f:
        docs = json.load(f)
    texts = [f"{d['title']}\n\n{d['body']}" for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    def stub(query, k=3, machine_type=None):
        candidates = list(range(len(docs)))
        if machine_type:
            candidates = [i for i in candidates if docs[i]["machine_type"] in (machine_type, "GENERAL")]
        if not candidates:
            return []
        q_vec = vectorizer.transform([query])
        sims = (matrix[candidates] @ q_vec.T).toarray().flatten()
        ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])[:k]
        return [{
            "text": f"{docs[i]['title']}\n\n{docs[i]['body']}",
            "doc_type": docs[i]["doc_type"], "machine_type": docs[i]["machine_type"],
            "title": docs[i]["title"], "distance": round(1 - sim, 3),
        } for i, sim in ranked]
    return stub


if __name__ == "__main__":
    chain_module.retrieve = _offline_retrieve_stub()
    fake_llm = FakeListChatModel(responses=["[FAKE LLM RESPONSE]"] * 20)
    assistant = ManufacturingAssistant(llm=fake_llm)

    report = run(assistant=assistant)
    print_report(report)
