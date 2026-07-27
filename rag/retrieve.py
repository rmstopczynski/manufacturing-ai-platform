"""
Reusable retrieval function against the persistent Chroma collection built by
build_index.py. Imported directly by the Phase 3 LangChain orchestration layer.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = "rag/chroma_db"
COLLECTION_NAME = "manufacturing_docs"

_embed_fn = None
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        # Must match the embedding function used in build_index.py exactly, or
        # query vectors and stored vectors won't be comparable.
        _embed_fn = embedding_functions.ONNXMiniLM_L6_V2()
    return _embed_fn


def get_collection():
    return _get_client().get_collection(name=COLLECTION_NAME, embedding_function=_get_embed_fn())


def retrieve(query: str, k: int = 3, machine_type: str | None = None) -> list[dict]:
    """Return the top-k most relevant chunks for a query, optionally filtered to a
    specific machine_type (plus GENERAL docs, which apply to all machine types)."""
    collection = get_collection()

    where = None
    if machine_type:
        where = {"machine_type": {"$in": [machine_type, "GENERAL"]}}

    results = collection.query(query_texts=[query], n_results=k, where=where)

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": text,
            "doc_type": meta["doc_type"],
            "machine_type": meta["machine_type"],
            "title": meta["title"],
            "distance": dist,
        })
    return chunks


if __name__ == "__main__":
    # Quick manual smoke test (requires internet access to download the embedding
    # model on first run — see README note on sandbox limitations).
    for r in retrieve("vibration climbing on a CNC mill", k=3, machine_type="CNC_MILL"):
        print(f"[{r['doc_type']}] {r['title']} (distance={r['distance']:.3f})")
