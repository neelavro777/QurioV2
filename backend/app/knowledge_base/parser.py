"""
parser.py — Reads each .txt knowledge base file and splits it into parent
sections by detecting Markdown '#' headings.

Contract:
  parse_file(path, file_key) -> list[ParentSection]

Each ParentSection contains the heading string, the full raw text of that
section (the "parent" used for LLM context), and the parent_id resolved
from headings_map.json.

The parser validates that every heading found in the file has a corresponding
entry in headings_map.json and raises clearly if not. This ensures the txt
files and the map never drift out of sync silently.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.knowledge_base.config import HEADINGS_MAP_PATH


@dataclass
class ParentSection:
    parent_id:    str
    heading:      str
    raw_text:     str          # full section text including heading line
    source_file:  str          # filename stem, e.g. "debit_card"
    product_area: str


def _load_headings_map() -> dict:
    with open(HEADINGS_MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_heading_to_parent_id(file_key: str, headings_map: dict) -> dict[str, str]:
    """
    Returns a dict mapping normalised heading string → parent_id for a given
    file key (e.g. "debit_card").
    """
    file_meta = headings_map.get(file_key)
    if file_meta is None:
        raise KeyError(
            f"File key '{file_key}' not found in headings_map.json. "
            f"Available keys: {list(headings_map.keys())}"
        )
    return {section["heading"]: section["parent_id"] for section in file_meta["sections"]}


def parse_file(path: Path, file_key: str, product_area: str) -> list[ParentSection]:
    """
    Split a knowledge-base .txt file into parent sections on '# Heading' lines.

    Args:
        path:         Absolute path to the .txt file.
        file_key:     Key into headings_map.json (e.g. "debit_card").
        product_area: Controlled vocabulary tag (e.g. "debit_card").

    Returns:
        Ordered list of ParentSection objects.

    Raises:
        KeyError:  If a heading found in the file has no mapping in headings_map.json.
        ValueError: If the file produces zero sections.
    """
    headings_map = _load_headings_map()
    heading_to_parent_id = _build_heading_to_parent_id(file_key, headings_map)

    raw = path.read_text(encoding="utf-8")
    source_file = path.stem

    # Split on lines that start with '# ' (level-1 heading only)
    # Pattern: find all heading positions and their text
    heading_pattern = re.compile(r"^# .+", re.MULTILINE)
    matches = list(heading_pattern.finditer(raw))

    if not matches:
        raise ValueError(
            f"No '# Heading' lines found in {path}. "
            "Ensure the file uses '# ' (hash + space) for section headings."
        )

    sections: list[ParentSection] = []

    for i, match in enumerate(matches):
        heading_line = match.group(0)                    # e.g. "# Card Delivery"
        heading_text = heading_line.lstrip("# ").strip() # e.g. "Card Delivery"

        # Section body: from end of this heading to start of next (or EOF)
        body_start = match.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        section_body = raw[body_start:body_end].strip()

        # Full parent text = heading + body (so the LLM gets the heading label)
        full_text = f"{heading_line}\n\n{section_body}"

        # Validate against headings_map
        if heading_text not in heading_to_parent_id:
            raise KeyError(
                f"Heading '{heading_text}' in file '{source_file}.txt' has no entry "
                f"in headings_map.json under key '{file_key}'. "
                f"Available headings: {list(heading_to_parent_id.keys())}"
            )

        parent_id = heading_to_parent_id[heading_text]

        sections.append(ParentSection(
            parent_id=parent_id,
            heading=heading_text,
            raw_text=full_text,
            source_file=source_file,
            product_area=product_area,
        ))

    return sections
