"""PDF ingestion, chunking, and metadata extraction for Fluent UDF documents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from tqdm import tqdm  # type: ignore[import-untyped]

SECTION_HEADER_RE = re.compile(r"^(\d+(?:\.\d+){0,4})\.?\s+([A-Z][^\n]{2,120})$")
DEFINE_MACRO_RE = re.compile(r"\bDEFINE_[A-Z0-9_]+\b")
CHARS_PER_TOKEN = 4  # rough proxy: 1 token ~= 4 chars for English prose


@dataclass
class Chunk:
    """Normalized chunk payload stored in JSONL and used for retrieval."""

    id: str
    source: str
    page_start: int
    page_end: int
    section: str
    macro_families: list[str]
    text: str
    token_estimate: int

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


@dataclass
class RawPage:
    """Extracted text for one PDF page."""

    page_number: int  # 1-indexed
    text: str


@dataclass
class _Accumulator:
    lines: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    section: str = "Unknown"

    def char_count(self) -> int:
        return sum(len(line) + 1 for line in self.lines)

    def add(self, line: str, page: int, section: str) -> None:
        if self.page_start is None:
            self.page_start = page
        self.page_end = page
        if section and section != "Unknown":
            self.section = section
        self.lines.append(line)

    def finalize(self, idx: int, source: str) -> Chunk | None:
        text = "\n".join(self.lines).strip()
        if not text:
            return None
        families = sorted(set(DEFINE_MACRO_RE.findall(text)))
        return Chunk(
            id=f"{Path(source).stem}::chunk_{idx:05d}",
            source=source,
            page_start=self.page_start or -1,
            page_end=self.page_end or -1,
            section=self.section,
            macro_families=families,
            text=text,
            token_estimate=max(1, len(text) // CHARS_PER_TOKEN),
        )


def extract_pages(pdf_path: Path) -> list[RawPage]:
    """Extract all pages from a PDF into raw text records."""
    reader = PdfReader(str(pdf_path))
    pages: list[RawPage] = []
    for i, page in enumerate(tqdm(reader.pages, desc=f"Parsing {pdf_path.name}")):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(RawPage(page_number=i + 1, text=text))
    return pages


def _detect_section(line: str, current: str) -> str:
    """Infer section headers from a line of manual text."""
    stripped = line.strip()
    if not stripped or len(stripped) > 140:
        return current
    match = SECTION_HEADER_RE.match(stripped)
    if match:
        number, title = match.group(1), match.group(2).strip()
        return f"{number} {title}"
    return current


def _iter_section_tagged_lines(
    pages: Iterable[RawPage],
) -> Iterable[tuple[int, str, str]]:
    """Yield non-empty lines tagged with their inferred section."""
    section = "Unknown"
    for raw in pages:
        for line in raw.text.splitlines():
            section = _detect_section(line, section)
            if line.strip():
                yield raw.page_number, section, line


def _section_depth(section: str) -> int:
    """Compute numeric depth from section labels like 2.3.1."""
    token = section.split(" ", 1)[0]
    return token.count(".") + 1 if token and token[0].isdigit() else 0


def chunk_pages(
    pages: list[RawPage],
    source: str,
    chunk_tokens: int = 200,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Split pages into overlap-aware chunks with section metadata."""
    target_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN
    min_chars_for_section_break = max(target_chars // 2, 400)

    chunks: list[Chunk] = []
    acc = _Accumulator()
    last_section = "Unknown"
    last_major = ""

    for page_number, section, line in _iter_section_tagged_lines(pages):
        section_changed = section != last_section and section != "Unknown"
        major = (
            ".".join(section.split(" ", 1)[0].split(".")[:2])
            if section != "Unknown"
            else ""
        )
        major_changed = bool(major) and bool(last_major) and major != last_major
        size_trigger = acc.char_count() >= target_chars
        section_trigger = (
            section_changed
            and acc.char_count() >= min_chars_for_section_break
            and (major_changed or _section_depth(section) <= 2)
        )
        if acc.lines and (size_trigger or section_trigger):
            finalized = acc.finalize(len(chunks), source)
            if finalized is not None:
                chunks.append(finalized)
            tail_lines: list[str] = []
            running = 0
            for prior in reversed(acc.lines):
                running += len(prior) + 1
                tail_lines.append(prior)
                if running >= overlap_chars:
                    break
            tail_lines.reverse()
            acc = _Accumulator(
                lines=list(tail_lines) if not section_trigger else [],
                page_start=page_number,
                page_end=page_number,
                section=section,
            )
        acc.add(line, page_number, section)
        last_section = section
        if major:
            last_major = major

    final = acc.finalize(len(chunks), source)
    if final is not None:
        chunks.append(final)
    return chunks


def chunk_text_file(
    path: Path, source: str, chunk_tokens: int, chunk_overlap: int
) -> list[Chunk]:
    """Chunk a plain-text source and tag chunks as example content."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    fake_page = RawPage(page_number=1, text=text)
    chunks = chunk_pages(
        [fake_page],
        source=source,
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    for chunk in chunks:
        chunk.section = f"Example: {path.name}"
    return chunks


def save_chunks(chunks: list[Chunk], out_path: Path | str) -> None:
    """Write chunk records as JSONL."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path | str) -> list[Chunk]:
    """Load chunk records from a JSONL file."""
    path = Path(path)
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks
