"""
sync.py
-------
Orchestrator script for the knowledge base synchronization pipeline.
Checks for stale text files using SHA-256 hashes and automatically
triggers ingestion and smart ChromaDB population if changes are detected.
"""

import hashlib
import json
import logging
from pathlib import Path

from app.knowledge_base.config import KNOWLEDGE_BASE_DIR, CHUNKS_OUTPUT_PATH, FILE_TO_PRODUCT_AREA
from app.knowledge_base.ingest import run_ingestion
from app.knowledge_base.populate_chroma import get_chroma_client, get_or_create_collection, populate_smart

log = logging.getLogger(__name__)

def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def run_sync():
    """
    1. Check for staleness using SHA-256 hashes.
    2. If stale, run ingestion to recreate chunks.json.
    3. Run smart chroma population to diff, embed new, and delete old chunks.
    """
    log.info("[sync] Starting knowledge base synchronization check...")
    
    # 1. Stale Detection
    current_hashes = {}
    for file_stem in FILE_TO_PRODUCT_AREA.keys():
        txt_path = KNOWLEDGE_BASE_DIR / f"{file_stem}.txt"
        if txt_path.exists():
            current_hashes[txt_path.name] = _compute_sha256(txt_path)
            
    total_chunks = 0
    is_stale = True
    if CHUNKS_OUTPUT_PATH.exists():
        try:
            with open(CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            stored_hashes = chunks_data.get("meta", {}).get("source_checksums", {})
            total_chunks = len(chunks_data.get("chunks", []))
            if stored_hashes == current_hashes:
                is_stale = False
        except Exception as e:
            log.warning("[sync] Failed to read chunks.json, treating as stale: %s", e)

    if not is_stale:
        log.info("[sync] Knowledge base is up-to-date. No changes detected.")
        print(f"  [KB] Knowledge base up-to-date — {total_chunks} chunks loaded.")
        return

    log.info("[sync] Changes detected in source files. Rebuilding knowledge base...")
    print("  [KB] Changes detected in source files. Synchronizing...")

    # 2. Ingestion (in memory)
    chunks_data = run_ingestion(save_to_disk=False)

    # 3. Smart ChromaDB Population
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    
    # Run smart population
    summary = populate_smart(collection, chunks_data)
    
    # 4. Commit to disk ONLY after ChromaDB succeeds
    with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"  [KB] Sync complete — {summary['upserted']} chunks added, {summary['deleted']} deleted.")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_sync()
