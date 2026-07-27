"""
SANDBOX-ONLY TEST — not part of the shipped pipeline.

Exercises the FastAPI app with TestClient. /predict is tested against the real
predictor (no external dependency needed). /ask overrides the get_assistant
dependency with a fake-LLM-backed assistant, since this sandbox can't reach
api.openai.com — this is exactly what FastAPI's dependency_overrides mechanism is
for, and is the same reasoning as swapping ChatOpenAI for FakeListChatModel in
orchestration/_sandbox_test_chain.py.
"""

import json
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sklearn.feature_extraction.text import TfidfVectorizer

import api.main as main_module
from api.main import app, get_assistant
from orchestration.chain import ManufacturingAssistant
import orchestration.chain as chain_module


def _offline_retrieve_stub():
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


def fake_assistant():
    chain_module.retrieve = _offline_retrieve_stub()
    fake_llm = FakeListChatModel(responses=["[FAKE LLM RESPONSE]"] * 20)
    return ManufacturingAssistant(llm=fake_llm)


app.dependency_overrides[get_assistant] = fake_assistant
client = TestClient(app)


def main():
    print("=== GET /health ===")
    r = client.get("/health")
    print(r.status_code, r.json())

    print("\n=== POST /predict?machine_id=M001 (known machine) ===")
    r = client.post("/predict", params={"machine_id": "M001"})
    print(r.status_code, r.json())

    print("\n=== POST /predict?machine_id=M999 (unknown machine) ===")
    r = client.post("/predict", params={"machine_id": "M999"})
    print(r.status_code, r.json())

    print("\n=== POST /predict?machine_id=   (empty) ===")
    r = client.post("/predict", params={"machine_id": "  "})
    print(r.status_code, r.json())

    print("\n=== POST /ask {question: 'Is M001 at risk?'} ===")
    r = client.post("/ask", json={"question": "Is machine M001 at risk of failing soon?"})
    print(r.status_code, json.dumps(r.json(), indent=2)[:800])

    print("\n=== POST /ask {question: ''} (validation) ===")
    r = client.post("/ask", json={"question": ""})
    print(r.status_code, r.json())


if __name__ == "__main__":
    main()
