"""
SANDBOX-ONLY TEST — not part of the shipped pipeline.

This environment's network can't reach huggingface.co to download the real
sentence-transformer model used in build_index.py. This script substitutes a
TF-IDF vectorizer as a stand-in embedding function purely to verify that the
chunking, Chroma storage, and retrieval logic all work correctly end to end.

Run `python3 rag/build_index.py` (the real script) once you have normal internet
access locally — it uses all-MiniLM-L6-v2 sentence embeddings as designed.
"""

import json
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, "rag")
from build_index import chunk_document, CHROMA_DIR  # reuse the real chunking logic

DOCS_MANIFEST = "rag/documents/manifest.json"


def main():
    with open(DOCS_MANIFEST) as f:
        documents = json.load(f)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts).toarray().tolist()

    client = chromadb.PersistentClient(path="/tmp/chroma_test_db")
    try:
        client.delete_collection("test_docs")
    except Exception:
        pass
    collection = client.create_collection(name="test_docs")  # no embedding_function -> we pass vectors directly

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=vectors,
        documents=texts,
        metadatas=[{
            "doc_id": c["doc_id"], "doc_type": c["doc_type"],
            "machine_type": c["machine_type"], "title": c["title"],
        } for c in all_chunks],
    )
    print(f"Indexed {collection.count()} chunks (TF-IDF stand-in embeddings, sandbox test only)\n")

    test_queries = [
        "vibration has been climbing on a CNC mill for a few weeks",
        "what should I do before doing maintenance on any machine",
        "pump bearing keeps failing",
        "how often should routine maintenance happen",
    ]

    for q in test_queries:
        q_vec = vectorizer.transform([q]).toarray().tolist()
        results = collection.query(query_embeddings=q_vec, n_results=3)
        print(f"QUERY: {q}")
        for rank, (meta, dist, text) in enumerate(zip(
            results["metadatas"][0], results["distances"][0], results["documents"][0]
        )):
            print(f"  #{rank+1} [{meta['doc_type']}/{meta['machine_type']}] "
                  f"{meta['title']} (distance={dist:.3f})")
        print()


if __name__ == "__main__":
    main()
