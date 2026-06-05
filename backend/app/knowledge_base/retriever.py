"""
retriever.py — ChromaDB-backed semantic retrieval for the Qurio knowledge base.

Core function:
    retrieve(query: str) -> RetrievalResult

Pipeline:
  1. Query ChromaDB collection for top-K candidate chunks by cosine similarity.
  2. Convert ChromaDB distance to similarity: similarity = 1 - distance
  3. Apply two-tier threshold filtering:
       - similarity < SIMILARITY_THRESHOLD_REJECT (0.35): hard discard (noise)
       - 0.35 <= similarity < SIMILARITY_THRESHOLD_WARN (0.55): low-confidence
       - similarity >= 0.55: high-confidence
  4. If ALL candidates are below threshold: return RetrievalResult(no_context=True)
     The caller must then use NO_CONTEXT_MESSAGE as the response.
  5. Deduplicate by parent_id: two children from the same parent section
     contribute only one parent context block.
  6. Expand to parent sections: for child chunks, look up the full parent text
     from chunks.json parents dict. For qa_augmented chunks (no parent_id),
     use the chunk text itself as the context.
  7. Assemble labelled context blocks:
     "[Context: {heading} | product: {product_area}]\n{text}"

Logging (every retrieval call):
  - Terminal: colour-coded confidence scores (green=high, yellow=low, red=reject)
  - logs/retrieval_trace.ndjson: one JSON line per retrieval call with:
      query, top_k_chunks (ids + scores + confidence), parents_expanded,
      product_areas, no_context, duration_ms

Cosine similarity thresholds (imported from config.py):
  SIMILARITY_THRESHOLD_REJECT = 0.35
  SIMILARITY_THRESHOLD_WARN   = 0.55
  See config.py for full calibration notes.

NO_CONTEXT fallback (imported from populate_chroma.py):
  NO_CONTEXT_MESSAGE = "I don't have information on that topic..."
  This is a constant, NOT an LLM generation — zero hallucination risk on
  the failure path, no added latency.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.knowledge_base.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_PATH,
    CHUNKS_OUTPUT_PATH,
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD_REJECT,
    SIMILARITY_THRESHOLD_WARN,
)
from app.knowledge_base.populate_chroma import NO_CONTEXT_MESSAGE, get_chroma_client

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NDJSON trace log path
# ---------------------------------------------------------------------------
_LOGS_DIR = Path(__file__).resolve().parents[3] / "logs"
_RETRIEVAL_TRACE_PATH = _LOGS_DIR / "retrieval_trace.ndjson"
_KB_LOGS_DIR = _LOGS_DIR / "knowledge_base"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
_KB_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ANSI colour helpers (terminal only — not in log files)
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"


def _colour_score(score: float) -> str:
    """Return ANSI-coloured similarity score string for terminal output."""
    if score >= SIMILARITY_THRESHOLD_WARN:
        return f"{_GREEN}{score:.4f}{_RESET}"
    elif score >= SIMILARITY_THRESHOLD_REJECT:
        return f"{_YELLOW}{score:.4f}{_RESET}"
    else:
        return f"{_RED}{score:.4f}{_RESET}"


def _confidence_label(score: float) -> str:
    if score >= SIMILARITY_THRESHOLD_WARN:
        return "high"
    elif score >= SIMILARITY_THRESHOLD_REJECT:
        return "low"
    else:
        return "reject"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """Represents one candidate chunk returned from ChromaDB."""
    chunk_id:   str
    text:       str
    distance:   float         # raw ChromaDB distance (cosine space)
    similarity: float         # 1 - distance
    confidence: str           # "high" | "low" | "reject"
    metadata:   dict = field(default_factory=dict)


@dataclass
class ContextBlock:
    """One parent section expanded from a matched child chunk."""
    parent_id:       Optional[str]
    heading:         str
    product_area:    str
    text:            str        # full parent text (or chunk text for qa_augmented)
    best_similarity: float      # highest similarity score among children of this parent
    supplementary_chunks: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """
    Complete output of one retrieve() call.

    If no_context=True, the caller MUST use NO_CONTEXT_MESSAGE as the response
    and must NOT pass an empty context to the LLM.
    """
    query:           str
    no_context:      bool
    context_blocks:  list[ContextBlock] = field(default_factory=list)
    raw_candidates:  list[RetrievedChunk] = field(default_factory=list)
    context_string:  str = ""           # ready-to-inject formatted context for the prompt
    duration_ms:     float = 0.0

    @property
    def has_context(self) -> bool:
        return not self.no_context and bool(self.context_blocks)


# ---------------------------------------------------------------------------
# Module-level singletons (lazy-loaded on first retrieve() call)
# ---------------------------------------------------------------------------
_client:     Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection]       = None
_parents:    Optional[dict]                      = None   # parent_id -> {heading, text, ...}


def _load_parents() -> dict:
    """Load the parents dict from chunks.json for context expansion."""
    with open(CHUNKS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("parents", {})


def _get_collection() -> chromadb.Collection:
    """Lazy-load and return the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        log.info("[retriever] Connecting to ChromaDB at %s", CHROMA_PERSIST_PATH)
        _client = get_chroma_client()
        ef = OpenAIEmbeddingFunction(
            api_key="sk-no-key-required",
            api_base="http://localhost:8001/v1",
            model_name=EMBEDDING_MODEL_NAME
        )
        _collection = _client.get_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=ef,
        )
        log.info(
            "[retriever] Collection '%s' loaded | %d docs",
            CHROMA_COLLECTION_NAME, _collection.count(),
        )
    return _collection


def _get_parents() -> dict:
    """Lazy-load and return the parents dict."""
    global _parents
    if _parents is None:
        _parents = _load_parents()
        log.info("[retriever] Loaded %d parent sections from chunks.json", len(_parents))
    return _parents


# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

def _write_trace(record: dict) -> None:
    """Append a JSON record to the retrieval trace NDJSON log."""
    try:
        with open(_RETRIEVAL_TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[retriever] Failed to write trace: %s", e)


def _write_detailed_log(session_id: str, record: dict) -> None:
    """Write a detailed JSON log file with full context to the knowledge_base directory."""
    short_id = session_id.replace("-", "")[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{short_id}_retrieval.json"
    filepath = _KB_LOGS_DIR / filename
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning("[retriever] Failed to write detailed KB log: %s", e)


def _print_retrieval_header(query: str) -> None:
    print(f"\n{'='*80}")
    print(f"{_CYAN}[RETRIEVER] Query: {query!r}{_RESET}")
    print(f"{'='*80}")


def _print_candidate(i: int, chunk: RetrievedChunk) -> None:
    score_str = _colour_score(chunk.similarity)
    conf_str  = (
        f"{_GREEN}HIGH{_RESET}"   if chunk.confidence == "high"   else
        f"{_YELLOW}LOW{_RESET}"   if chunk.confidence == "low"    else
        f"{_RED}REJECT{_RESET}"
    )
    print(
        f"  [{i+1}] similarity={score_str} | conf={conf_str} | "
        f"chunk_id={chunk.chunk_id}"
    )
    print(f"       text={chunk.text[:100]!r}")


def _print_retrieval_summary(result: RetrievalResult) -> None:
    if result.no_context:
        print(f"  {_RED}→ NO_CONTEXT: all candidates below threshold{_RESET}")
    else:
        print(f"  {_GREEN}→ {len(result.context_blocks)} parent section(s) expanded{_RESET}")
        for cb in result.context_blocks:
            print(
                f"     • {cb.parent_id or '(augmented)'} | "
                f"{cb.product_area} | sim={cb.best_similarity:.4f}"
            )
    print(f"  Duration: {result.duration_ms:.1f}ms")
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve(query: str, session_id: Optional[str] = None) -> RetrievalResult:
    """
    Retrieve the most relevant knowledge base context for a user query.

    Args:
        query: The raw user query string.

    Returns:
        RetrievalResult with context_blocks assembled from parent sections,
        a ready-to-inject context_string for the prompt, and full trace data.
        If no relevant context is found, no_context=True and context_string is empty.

    Raises:
        RuntimeError: If ChromaDB collection is not populated (call populate_chroma first).
    """
    start_time = time.monotonic()

    _print_retrieval_header(query)
    log.info("[retriever] Query: %r", query)

    # ── 1. ChromaDB similarity search ─────────────────────────────────────────
    try:
        collection = _get_collection()
    except Exception as e:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' not available. "
            f"Run python -m app.knowledge_base.populate_chroma first. Error: {e}"
        ) from e

    try:
        results = collection.query(
            query_texts=[query],
            n_results=RETRIEVAL_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.error("[retriever] ChromaDB query failed: %s", e)
        elapsed = round((time.monotonic() - start_time) * 1000, 1)
        return RetrievalResult(
            query=query, no_context=True, duration_ms=elapsed
        )

    # ── 2. Parse and score candidates ─────────────────────────────────────────
    ids       = results["ids"][0]          # list of chunk_ids
    documents = results["documents"][0]    # list of chunk texts
    metadatas = results["metadatas"][0]    # list of metadata dicts
    distances = results["distances"][0]    # list of cosine distances

    print(f"  {_CYAN}Top-{RETRIEVAL_TOP_K} candidates:{_RESET}")

    candidates: list[RetrievedChunk] = []
    for i, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances)
    ):
        similarity = 1.0 - dist
        confidence = _confidence_label(similarity)
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            distance=dist,
            similarity=similarity,
            confidence=confidence,
            metadata=meta,
        )
        candidates.append(chunk)
        _print_candidate(i, chunk)

    # ── 3. Filter by threshold ─────────────────────────────────────────────────
    accepted = [c for c in candidates if c.confidence in ("high", "low")]

    if not accepted:
        elapsed = round((time.monotonic() - start_time) * 1000, 1)
        log.info(
            "[retriever] NO_CONTEXT: all %d candidates below threshold %.2f",
            len(candidates), SIMILARITY_THRESHOLD_REJECT,
        )

        result = RetrievalResult(
            query=query,
            no_context=True,
            raw_candidates=candidates,
            duration_ms=elapsed,
        )
        _print_retrieval_summary(result)
        _write_trace({
            "event":          "retrieval",
            "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query":          query,
            "no_context":     True,
            "top_k_chunks":   [
                {
                    "chunk_id":   c.chunk_id,
                    "similarity": round(c.similarity, 4),
                    "confidence": c.confidence,
                    "parent_id":  c.metadata.get("parent_id"),
                }
                for c in candidates
            ],
            "parents_expanded": [],
            "product_areas":  [],
            "duration_ms":    elapsed,
        })
        
        if session_id:
            _write_detailed_log(session_id, {
                "event":          "retrieval",
                "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "session_id":     session_id,
                "query":          query,
                "no_context":     True,
                "top_k_chunks":   [
                    {
                        "chunk_id":   c.chunk_id,
                        "similarity": round(c.similarity, 4),
                        "confidence": c.confidence,
                        "parent_id":  c.metadata.get("parent_id"),
                        "text":       c.text,
                    }
                    for c in candidates
                ],
                "parents_expanded": [],
                "product_areas":  [],
                "context_string": "",
                "duration_ms":    elapsed,
            })
            
        return result

    # ── 4. Deduplicate by parent_id and expand to parent sections ────────────
    parents = _get_parents()

    seen_parent_ids: dict[str, ContextBlock] = {}
    context_blocks: list[ContextBlock] = []

    # Sort accepted chunks by similarity descending so best matches expand first
    accepted_sorted = sorted(accepted, key=lambda c: c.similarity, reverse=True)

    for chunk in accepted_sorted:
        parent_id    = chunk.metadata.get("parent_id") or ""
        chunk_type   = chunk.metadata.get("chunk_type", "child")
        product_area = chunk.metadata.get("product_area", "unknown")
        heading      = chunk.metadata.get("section_heading", "Unknown Section")

        if chunk_type == "qa_augmented":
            # Augmented chunks have no parent — the chunk text IS the context
            block_key = chunk.chunk_id   # deduplicate by chunk_id
            if block_key in seen_parent_ids:
                seen_parent_ids[block_key].supplementary_chunks.append(chunk.text)
                continue
            
            cb = ContextBlock(
                parent_id=None,
                heading=heading,
                product_area=product_area,
                text=chunk.text,
                best_similarity=chunk.similarity,
            )
            seen_parent_ids[block_key] = cb
            context_blocks.append(cb)
        else:
            # Child chunk — expand to full parent section
            if parent_id in seen_parent_ids:
                seen_parent_ids[parent_id].supplementary_chunks.append(chunk.text)
                continue

            parent_data = parents.get(parent_id)
            if parent_data:
                parent_text   = parent_data["text"]
                parent_heading = parent_data.get("heading", heading)
                parent_area    = parent_data.get("product_area", product_area)
            else:
                # Parent not found in index — fall back to chunk text
                log.warning(
                    "[retriever] parent_id %r not found in parents index, "
                    "falling back to chunk text", parent_id,
                )
                parent_text    = chunk.text
                parent_heading = heading
                parent_area    = product_area

            cb = ContextBlock(
                parent_id=parent_id,
                heading=parent_heading,
                product_area=parent_area,
                text=parent_text,
                best_similarity=chunk.similarity,
            )
            seen_parent_ids[parent_id] = cb
            context_blocks.append(cb)

    # ── 5. Assemble context string ─────────────────────────────────────────────
    context_parts: list[str] = []
    for cb in context_blocks:
        label = f"[Context: {cb.heading} | product: {cb.product_area}]"
        
        if cb.supplementary_chunks:
            supp_text = "\n\nHighlighted Relevant Excerpts (from same section):\n- " + \
                        "\n- ".join([c.replace("\n", " ") for c in cb.supplementary_chunks])
            context_parts.append(f"{label}\n{cb.text}{supp_text}")
        else:
            context_parts.append(f"{label}\n{cb.text}")

    context_string = "\n\n---\n\n".join(context_parts)

    elapsed = round((time.monotonic() - start_time) * 1000, 1)

    result = RetrievalResult(
        query=query,
        no_context=False,
        context_blocks=context_blocks,
        raw_candidates=candidates,
        context_string=context_string,
        duration_ms=elapsed,
    )

    _print_retrieval_summary(result)

    # ── 6. Write trace record ──────────────────────────────────────────────────
    _write_trace({
        "event":     "retrieval",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query":     query,
        "no_context": False,
        "top_k_chunks": [
            {
                "chunk_id":   c.chunk_id,
                "similarity": round(c.similarity, 4),
                "confidence": c.confidence,
                "parent_id":  c.metadata.get("parent_id"),
                "accepted":   c.confidence in ("high", "low"),
            }
            for c in candidates
        ],
        "parents_expanded": [
            cb.parent_id for cb in context_blocks
        ],
        "product_areas": list({cb.product_area for cb in context_blocks}),
        "context_block_count": len(context_blocks),
        "duration_ms": elapsed,
    })

    if session_id:
        _write_detailed_log(session_id, {
            "event":     "retrieval",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": session_id,
            "query":     query,
            "no_context": False,
            "top_k_chunks": [
                {
                    "chunk_id":   c.chunk_id,
                    "similarity": round(c.similarity, 4),
                    "confidence": c.confidence,
                    "parent_id":  c.metadata.get("parent_id"),
                    "accepted":   c.confidence in ("high", "low"),
                    "text":       c.text,
                }
                for c in candidates
            ],
            "parents_expanded": [
                {
                    "parent_id": cb.parent_id,
                    "heading": cb.heading,
                    "product_area": cb.product_area,
                    "best_similarity": round(cb.best_similarity, 4),
                    "text": cb.text,
                    "supplementary_chunks": cb.supplementary_chunks,
                } for cb in context_blocks
            ],
            "product_areas": list({cb.product_area for cb in context_blocks}),
            "context_block_count": len(context_blocks),
            "context_string": context_string,
            "duration_ms": elapsed,
        })

    log.info(
        "[retriever] Retrieved %d parent section(s) | %.1fms | "
        "products: %s",
        len(context_blocks),
        elapsed,
        [cb.product_area for cb in context_blocks],
    )

    return result


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Smoke test the knowledge base retriever.")
    parser.add_argument("--query", type=str, default="what is the ATM withdrawal limit?")
    args = parser.parse_args()

    print(f"\nRunning retriever smoke test...")
    print(f"Query: {args.query!r}\n")

    result = retrieve(args.query)

    if result.no_context:
        print(f"\nFallback message:\n  {NO_CONTEXT_MESSAGE}")
    else:
        print(f"\nContext assembled ({len(result.context_blocks)} blocks):")
        print("-" * 60)
        print(result.context_string[:500] + "..." if len(result.context_string) > 500 else result.context_string)
