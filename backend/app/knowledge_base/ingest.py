"""
ingest.py — Orchestrates the full knowledge base ingestion pipeline.

Pipeline:
  1. For each source .txt file, call parser.parse_file() to produce
     ParentSection objects (one per '# Heading' section).
  2. Call chunker.chunk_sections() to split each parent into child chunks
     with full metadata attached.
  3. Serialize all chunks to chunks.json in this package directory.

This script can be run directly:
    python -m app.knowledge_base.ingest

Or imported by future modules (e.g. ChromaDB population) that need the
processed chunk list without re-reading the source files.

Output (chunks.json) schema:
  {
    "meta": {
      "total_parents": int,
      "total_chunks":  int,
      "files_processed": list[str],
    },
    "parents": {
      "<parent_id>": {
        "heading":      str,
        "source_file":  str,
        "product_area": str,
        "text":         str,   # full parent text for LLM context expansion
      }, ...
    },
    "chunks": [
      {
        "chunk_id":  str,
        "text":      str,
        "metadata":  { ...full schema... },
        "parent_id": str,     # use this to look up parent text at query time
      }, ...
    ]
  }
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

from app.knowledge_base.config import (
    CHUNKS_OUTPUT_PATH,
    FILE_TO_PRODUCT_AREA,
    KNOWLEDGE_BASE_DIR,
)
from app.knowledge_base.parser import ParentSection, parse_file
from app.knowledge_base.chunker import Chunk, chunk_sections

log = logging.getLogger(__name__)


def _serialize_parent(section: ParentSection) -> dict:
    return {
        "heading":      section.heading,
        "source_file":  section.source_file,
        "product_area": section.product_area,
        "text":         section.raw_text,
    }


def _serialize_chunk(chunk: Chunk) -> dict:
    return {
        "chunk_id":  chunk.chunk_id,
        "text":      chunk.text,
        "metadata":  chunk.metadata,
        "parent_id": chunk.metadata["parent_id"],
    }


def run_ingestion(save_to_disk: bool = True) -> dict:
    """
    Run the full parse → chunk → save pipeline for all 7 knowledge base files.

    Returns:
        The full output dict that was written to chunks.json.
    """
    all_sections: list[ParentSection] = []
    all_chunks:   list[Chunk]         = []
    files_processed: list[str]        = []
    source_checksums: dict[str, str]  = {}

    headings_map_path = KNOWLEDGE_BASE_DIR / "headings_map.json"
    headings_map = {}
    if headings_map_path.exists():
        with open(headings_map_path, "r", encoding="utf-8") as f:
            headings_map = json.load(f)

    for file_stem, product_area in FILE_TO_PRODUCT_AREA.items():
        txt_path = KNOWLEDGE_BASE_DIR / f"{file_stem}.txt"

        if not txt_path.exists():
            log.warning("Source file not found, skipping: %s", txt_path)
            continue

        log.info("Parsing: %s  →  product_area='%s'", txt_path.name, product_area)

        sections = parse_file(
            path=txt_path,
            file_key=file_stem,
            product_area=product_area,
        )
        chunks = chunk_sections(sections)

        log.info(
            "  %s: %d section(s), %d chunk(s)",
            file_stem, len(sections), len(chunks),
        )

        for chunk in chunks:
            if len(chunk.text) < 80:
                log.warning(
                    "Tiny chunk detected (%d chars) in %s: %r",
                    len(chunk.text), file_stem, chunk.text
                )

        all_sections.extend(sections)
        all_chunks.extend(chunks)
        files_processed.append(txt_path.name)
        source_checksums[txt_path.name] = _compute_sha256(txt_path)

    parsed_headings = {section.heading for section in all_sections}
    for file_key, file_data in headings_map.items():
        if file_key in FILE_TO_PRODUCT_AREA:
            for s in file_data.get("sections", []):
                h = s.get("heading")
                if h and h not in parsed_headings:
                    log.warning(
                        "Heading %r from headings_map.json not found in parsed sections of %s",
                        h, file_key
                    )

    # Build parents index (keyed by parent_id for O(1) lookup at query time)
    parents_index: dict[str, dict] = {
        section.parent_id: _serialize_parent(section)
        for section in all_sections
    }

    output = {
        "meta": {
            "total_parents":   len(all_sections),
            "total_chunks":    len(all_chunks),
            "files_processed": files_processed,
            "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_checksums": source_checksums,
        },
        "parents": parents_index,
        "chunks":  [_serialize_chunk(c) for c in all_chunks],
    }

    if save_to_disk:
        CHUNKS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info(
            "Ingestion complete: %d parents, %d chunks → %s",
            len(all_sections), len(all_chunks), CHUNKS_OUTPUT_PATH,
        )
    else:
        log.info(
            "Ingestion complete in memory: %d parents, %d chunks",
            len(all_sections), len(all_chunks)
        )
    return output


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run_ingestion()
    meta = result["meta"]
    print(f"\n✓ Done. {meta['total_parents']} parents | {meta['total_chunks']} chunks")
    print(f"  Saved to: {CHUNKS_OUTPUT_PATH}")
