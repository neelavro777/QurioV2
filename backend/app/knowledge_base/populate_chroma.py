"""
populate_chroma.py — Reads chunks.json and populates the ChromaDB collection.

Design decisions:
  - PersistentClient at backend/chroma_db/ (disk-persisted, embedded in-process).
  - Collection name: qurio_knowledge_v1 (versioned for schema migrations).
  - Distance function: cosine (HNSW space="cosine").
  - Embedding model: BAAI/bge-small-en-v1.5 via SentenceTransformerEmbeddingFunction.
    English-only, 33M params, fast. Upgrade to multilingual-e5-small for Bengali support.
  - Stores ONLY chunk_id, text, and metadata in ChromaDB.
    parent_text is NOT stored (would bloat the index) — fetched at query time
    from the parents dict in chunks.json.
  - Uses .upsert() (not .add()) so re-running this script is safe and idempotent.
    Same chunk_id + same text = no-op. Changed text = updated vector.
  - metadata["topics"] is a list — ChromaDB does not support list metadata values.
    It is serialized as a comma-joined string for storage and must be deserialized
    on retrieval if needed.

Run this after ingest.py and optionally augment.py:
    python -m app.knowledge_base.ingest
    python -m app.knowledge_base.augment   # optional but recommended
    python -m app.knowledge_base.populate_chroma

The collection will be at: backend/chroma_db/
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.knowledge_base.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_PATH,
    CHUNKS_OUTPUT_PATH,
    EMBEDDING_MODEL_NAME,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NO_CONTEXT fallback message — defined here as the canonical source of truth.
# The retriever imports this constant so the string is never duplicated.
# ---------------------------------------------------------------------------
NO_CONTEXT_MESSAGE = (
    "I don't have information on that topic in my knowledge base. "
    "For assistance, please contact Prime NOW support at 09644-213030 "
    "or 09610-961000, or reach out through the in-app chat."
)


def _sanitize_metadata(metadata: dict) -> dict:
    """
    ChromaDB metadata values must be str, int, float, or bool.
    Lists (e.g. topics) are not supported — serialize them as comma-joined strings.
    None values are also not supported — replace with empty string.
    """
    sanitized = {}
    for k, v in metadata.items():
        if v is None:
            sanitized[k] = ""
        elif isinstance(v, list):
            # Join list items as comma-separated string for ChromaDB storage.
            # When reading back, split on "," to reconstruct the list.
            sanitized[k] = ",".join(str(x) for x in v)
        else:
            sanitized[k] = v
    return sanitized


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a PersistentClient pointed at the project's chroma_db directory.
    Creates the directory if it doesn't exist.
    """
    CHROMA_PERSIST_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_PATH))


def get_or_create_collection(
    client: chromadb.PersistentClient,
) -> chromadb.Collection:
    """
    Get or create the knowledge base collection with cosine similarity distance
    and the bge-small-en-v1.5 embedding function.

    HNSW space="cosine" means ChromaDB returns distances where:
      distance = 1 - cosine_similarity
    So: cosine_similarity = 1 - distance
    A distance of 0 = identical vectors (perfect match).
    A distance of 1 = orthogonal vectors (completely different).
    """
    ef = OpenAIEmbeddingFunction(
        api_key="sk-no-key-required",
        api_base="http://localhost:8001/v1",
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    log.info(
        "[populate_chroma] Collection '%s' ready | existing docs: %d",
        CHROMA_COLLECTION_NAME, collection.count(),
    )
    return collection


def populate_smart(collection: chromadb.Collection, chunks_data: dict) -> dict:
    """
    Smart upsert: compares chunk_ids in chunks.json with those in ChromaDB.
    Only computes embeddings and upserts for NEW or CHANGED chunks.
    Deletes stale chunks that no longer exist in the source files.
    """
    chunks = chunks_data["chunks"]
    total_in_file = len(chunks)
    log.info("[populate_chroma] Smart sync for %d chunks...", total_in_file)

    start_time = time.monotonic()

    # Get existing IDs from ChromaDB
    existing_ids = set(collection.get()["ids"])
    
    # Target IDs
    target_ids = {chunk["chunk_id"] for chunk in chunks}
    
    # IDs to delete
    ids_to_delete = list(existing_ids - target_ids)
    
    # IDs to add/embed
    ids_to_add = target_ids - existing_ids
    
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        log.info("[populate_chroma] Deleted %d stale chunks.", len(ids_to_delete))

    # Filter chunks to only those that need to be added
    new_chunks = [c for c in chunks if c["chunk_id"] in ids_to_add]
    
    BATCH_SIZE = 100
    upserted = 0
    skipped = 0

    if new_chunks:
        log.info("[populate_chroma] Computing embeddings and upserting %d new chunks...", len(new_chunks))
        for batch_start in range(0, len(new_chunks), BATCH_SIZE):
            batch = new_chunks[batch_start : batch_start + BATCH_SIZE]
            
            ids = []
            documents = []
            metadatas = []

            for chunk in batch:
                if not chunk.get("chunk_id") or not chunk.get("text"):
                    log.warning(
                        "[populate_chroma] Skipping chunk with missing id or text: %s",
                        chunk.get("chunk_id", "<no id>"),
                    )
                    skipped += 1
                    continue

                ids.append(chunk["chunk_id"])
                documents.append(chunk["text"])
                metadatas.append(_sanitize_metadata(chunk["metadata"]))

            if ids:
                # If embedding server is down, this will intentionally raise an Exception
                # causing the app to fail to start, as requested by user.
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                upserted += len(ids)
                log.info(
                    "[populate_chroma]   Batch %d-%d: upserted %d chunks",
                    batch_start + 1, batch_start + len(batch), len(ids),
                )

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 1)

    summary = {
        "event":          "chroma_population_complete",
        "collection":     CHROMA_COLLECTION_NAME,
        "total_in_file":  total_in_file,
        "upserted":       upserted,
        "deleted":        len(ids_to_delete),
        "skipped":        skipped,
        "final_count":    collection.count(),
        "duration_ms":    elapsed_ms,
        "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "persist_path":   str(CHROMA_PERSIST_PATH),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

    log.info(
        "[populate_chroma] Sync Done. %d added, %d deleted | "
        "Collection has %d docs | %.1fms",
        upserted, len(ids_to_delete), summary["final_count"], elapsed_ms,
    )

    return summary


def run_population() -> dict:
    """
    Full population pipeline: read chunks.json → connect to ChromaDB → upsert.

    Returns:
        Summary dict from populate_collection().
    """
    if not CHUNKS_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_OUTPUT_PATH}. "
            "Run python -m app.knowledge_base.ingest first."
        )

    with open(CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    meta = chunks_data.get("meta", {})
    log.info(
        "[populate_chroma] Loading chunks.json | "
        "generated_at=%s | parents=%d | chunks=%d",
        meta.get("generated_at", "unknown"),
        meta.get("total_parents", "?"),
        meta.get("total_chunks", "?"),
    )

    client     = get_chroma_client()
    collection = get_or_create_collection(client)
    summary    = populate_smart(collection, chunks_data)

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run_population()
    print(f"\n✓ ChromaDB population complete.")
    print(f"  Collection     : {result['collection']}")
    print(f"  Upserted       : {result['upserted']}")
    print(f"  Skipped        : {result['skipped']}")
    print(f"  Total in DB    : {result['final_count']}")
    print(f"  Duration       : {result['duration_ms']}ms")
    print(f"  Persist path   : {result['persist_path']}")
    print(f"  Embedding model: {result['embedding_model']}")
