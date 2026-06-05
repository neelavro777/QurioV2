"""
chunker.py — Splits each ParentSection into child chunks using
RecursiveCharacterTextSplitter, then attaches the full metadata schema
to every chunk.

Metadata schema (every chunk carries all fields):
  - source_file:     str   — filename stem, e.g. "debit_card"
  - product_area:    str   — controlled vocabulary, used in ChromaDB where-filter
  - section_heading: str   — human-readable heading label for prompt assembly
  - chunk_type:      str   — "child" (paragraph split) or "qa_pair" (augmentation)
  - parent_id:       str   — foreign key linking child → parent section text
  - topics:          list  — hand-curated keyword tags (debug only in v1)

Output shape (per chunk):
  {
    "chunk_id":   str,      # "{parent_id}__c{index}"
    "text":       str,      # the child chunk text sent to the embedder
    "metadata":   dict,     # the full schema above
    "parent_text": str,     # the full parent section text (fetched at query time)
  }
"""

import hashlib
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.knowledge_base.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_SEPARATORS,
    PARENT_TOPICS,
)
from app.knowledge_base.parser import ParentSection


@dataclass
class Chunk:
    chunk_id:    str
    text:        str
    metadata:    dict
    parent_text: str


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=CHILD_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True,
    )


def chunk_parent_section(
    section: ParentSection,
    chunk_type: str = "child",
) -> list[Chunk]:
    """
    Split a ParentSection into child chunks and attach metadata to each.

    Args:
        section:    A parsed ParentSection with heading, raw_text, etc.
        chunk_type: "child" for normal paragraph splits, "qa_pair" for
                    augmentation entries added in a second pass.

    Returns:
        List of Chunk objects, each carrying its full metadata dict and
        a reference to the full parent_text for context expansion at query time.
    """
    splitter = _make_splitter()

    # Strip the leading "# Heading" line before splitting.
    # The heading is already captured in section.heading / metadata["section_heading"].
    # Keeping it causes a 30-40 char heading-only orphan chunk that embeds poorly
    # and pollutes retrieval results.
    lines = section.raw_text.split("\n")
    body_lines = [l for l in lines if not l.startswith("# ")]
    body_text = "\n".join(body_lines).strip()

    raw_chunks: list[str] = splitter.split_text(body_text)

    topics = PARENT_TOPICS.get(section.parent_id, [])

    chunks: list[Chunk] = []
    for idx, text in enumerate(raw_chunks):
        text = text.strip()
        if not text:
            continue

        md5_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{section.parent_id}__{md5_hash}"

        metadata = {
            "source_file":     section.source_file,
            "product_area":    section.product_area,
            "section_heading": section.heading,
            "chunk_type":      chunk_type,
            "parent_id":       section.parent_id,
            "topics":          topics,
        }

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
            parent_text=section.raw_text,
        ))

    return chunks


def chunk_sections(sections: list[ParentSection]) -> list[Chunk]:
    """
    Chunk all parent sections from a single file.

    Args:
        sections: Output of parser.parse_file().

    Returns:
        Flat list of all child Chunk objects across all sections.
    """
    all_chunks: list[Chunk] = []
    for section in sections:
        all_chunks.extend(chunk_parent_section(section, chunk_type="child"))
    return all_chunks
