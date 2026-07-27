"""
Chunks the synthetic documents, embeds each chunk with a sentence-transformer model,
and loads them into a persistent ChromaDB collection.

Chunking strategy (worth defending explicitly, since "how did you decide on your
chunking strategy" is a near-guaranteed interview question):

These documents are SHORT by design (typically 100-250 words each — a manual section,
an SOP, or a maintenance log entry). At this scale, splitting a document into multiple
chunks would often cut a single warning-sign explanation or procedure step in half,
which actively hurts retrieval quality rather than helping it. So the chunking unit
here is "one document = one chunk" for anything under ~300 words, with a real
character-based splitter (kept in the code, not just described) that would kick in
for anything longer.

This is a genuine, defensible choice at THIS scale, but it does not scale to a real
enterprise deployment with long multi-page manuals — see the README design-decisions
note on what changes at larger document scale (fixed-size overlapping chunks, or
section-aware splitting on manual headers, become necessary once documents exceed a
page or two).

Output: a persistent Chroma collection at rag/chroma_db/, plus a retrieval function
importable by the LangChain orchestration layer (Phase 3).
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = "rag/documents"
MANIFEST_PATH = os.path.join(DOCS_DIR, "manifest.json")
CHROMA_DIR = "rag/chroma_db"
COLLECTION_NAME = "manufacturing_docs"

# Chunk size threshold, in characters. Below this, a document is kept as a single
# chunk (see chunking strategy note above). Above it, split with overlap so no
# sentence gets cut in a way that destroys its meaning.
CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150


def simple_overlapping_split(text: str, size: int, overlap: int) -> list[str]:
    """Character-based splitter with overlap, used only for documents that exceed
    CHUNK_SIZE_CHARS. Not exercised by the current synthetic corpus (all documents
    are short), but included because a real deployment's manuals won't be."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    full_text = f"{doc['title']}\n\n{doc['body']}"
    if len(full_text) <= CHUNK_SIZE_CHARS:
        return [{
            "chunk_id": f"{doc['doc_id']}_chunk0",
            "text": full_text,
            "doc_id": doc["doc_id"],
            "doc_type": doc["doc_type"],
            "machine_type": doc["machine_type"],
            "title": doc["title"],
        }]
    raw_chunks = simple_overlapping_split(full_text, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    return [{
        "chunk_id": f"{doc['doc_id']}_chunk{i}",
        "text": c,
        "doc_id": doc["doc_id"],
        "doc_type": doc["doc_type"],
        "machine_type": doc["machine_type"],
        "title": doc["title"],
    } for i, c in enumerate(raw_chunks)]


def build_index():
    with open(MANIFEST_PATH) as f:
        documents = json.load(f)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    print(f"{len(documents)} documents -> {len(all_chunks)} chunks "
          f"(all single-chunk: {len(all_chunks) == len(documents)})")

    # sentence-transformers/all-MiniLM-L6-v2: 384-dim, fast, well-understood baseline
    # for semantic similarity — no need for a larger model at this corpus size (24 docs).
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    # Fresh collection each run, so re-running this script after editing documents
    # doesn't leave stale chunks behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{
            "doc_id": c["doc_id"],
            "doc_type": c["doc_type"],
            "machine_type": c["machine_type"],
            "title": c["title"],
        } for c in all_chunks],
    )

    print(f"Indexed {collection.count()} chunks into Chroma collection '{COLLECTION_NAME}' "
          f"at {CHROMA_DIR}")
    return collection


if __name__ == "__main__":
    build_index()
