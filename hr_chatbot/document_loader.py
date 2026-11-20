"""DOCX manual loading and structure-aware chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

from hr_chatbot.schemas import HRRole


MANUAL_FILES: dict[HRRole, str] = {
    HRRole.EMPLOYEE: "ERP_HR_Xodim_qollanma.docx",
    HRRole.DEPARTMENT_MANAGER: "ERP_HR_HRManager_qollanma.docx",
    HRRole.BRANCH_MANAGER: "ERP_HR_BranchManager_qollanma.docx",
    HRRole.OWNER: "ERP_HR_Owner_qollanma.docx",
}


@dataclass(frozen=True, slots=True)
class ManualChunk:
    """A searchable fragment from one role manual."""

    chunk_id: str
    role: HRRole
    document: str
    text: str
    source_hash: str
    title_path: str = ""


@dataclass(frozen=True, slots=True)
class _DocumentBlock:
    text: str
    heading_level: int | None = None


@dataclass(frozen=True, slots=True)
class _SemanticSection:
    title_path: str
    header_lines: tuple[str, ...]
    body_lines: tuple[str, ...]
    # Bo'lim ichidagi sarlavhalar ("Bo'limni qanday ochish", "Yangi ... qo'shish").
    # Bo'lim uzun bo'lsa, kesish shu chegaralardan o'tadi.
    subheadings: frozenset[str] = frozenset()


def manuals_fingerprint(manuals_dir: Path) -> dict[str, str]:
    """Return file hashes used to detect stale vector stores."""

    fingerprints: dict[str, str] = {}
    for role, filename in MANUAL_FILES.items():
        path = manuals_dir / filename
        if path.exists():
            fingerprints[role.value] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def load_manual_chunks(manuals_dir: Path, *, chunk_size: int, chunk_overlap: int) -> list[ManualChunk]:
    """Read manuals and split them by their actual Word heading hierarchy."""

    chunks: list[ManualChunk] = []
    for role, filename in MANUAL_FILES.items():
        path = manuals_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Manual file not found: {path}")

        blocks = _read_docx_blocks(path)
        sections = _build_semantic_sections(blocks)
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        index = 1
        for section in sections:
            section_chunks = _split_semantic_section(
                section,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk_text in section_chunks:
                chunks.append(
                    ManualChunk(
                        chunk_id=f"{role.value}-{index:04d}",
                        role=role,
                        document=filename,
                        text=chunk_text,
                        source_hash=source_hash,
                        title_path=section.title_path,
                    )
                )
                index += 1
    return chunks


def _read_docx_blocks(path: Path) -> list[_DocumentBlock]:
    """Read paragraphs and tables in the same order in which they occur in Word."""

    document = Document(path)
    blocks: list[_DocumentBlock] = []
    for item in _iter_document_items(document):
        if isinstance(item, Paragraph):
            text = " ".join(item.text.split())
            if text:
                blocks.append(_DocumentBlock(text=text, heading_level=_heading_level(item)))
            continue

        for row in item.rows:
            values = [" ".join(cell.text.split()) for cell in row.cells]
            if any(values):
                blocks.append(_DocumentBlock(text=" | ".join(values)))
    return blocks


def _iter_document_items(document: DocumentObject) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = paragraph.style.name if paragraph.style else ""
    match = re.match(r"^Heading\s+(\d+)$", style_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _build_semantic_sections(blocks: list[_DocumentBlock]) -> list[_SemanticSection]:
    """Use Heading 1 for flat manuals and Heading 2 for deeply nested manuals."""

    if not blocks:
        return []

    # Xodim uses H1 as a feature and H2 as its details. The other manuals use
    # H1 as a broad group, H2 as a feature and H3 as feature details.
    topic_level = 2 if any(block.heading_level == 3 for block in blocks) else 1
    sections: list[_SemanticSection] = []
    parent_headings: dict[int, str] = {}
    current_title = ""
    current_headers: list[str] = []
    current_body: list[str] = []
    current_subheadings: set[str] = set()

    def flush() -> None:
        nonlocal current_title, current_headers, current_body, current_subheadings
        if current_headers or current_body:
            title_path = current_title or "Qo'llanma haqida"
            sections.append(
                _SemanticSection(
                    title_path=title_path,
                    header_lines=tuple(current_headers),
                    body_lines=tuple(current_body),
                    subheadings=frozenset(current_subheadings),
                )
            )
        current_title = ""
        current_headers = []
        current_body = []
        current_subheadings = set()

    for block in blocks:
        level = block.heading_level
        if level is not None and level < topic_level:
            flush()
            parent_headings[level] = block.text
            parent_headings = {key: value for key, value in parent_headings.items() if key <= level}
            current_title = block.text
            current_headers = [block.text]
            continue

        if level == topic_level:
            flush()
            parents = [parent_headings[key] for key in sorted(parent_headings) if key < topic_level]
            current_title = " > ".join([*parents, block.text])
            current_headers = [f"Asosiy bo'lim: {parent}" for parent in parents]
            current_headers.append(f"Mavzu: {block.text}")
            continue

        if level is not None and level > topic_level:
            current_subheadings.add(block.text)
        current_body.append(block.text)

    flush()
    return sections


def _split_semantic_section(
    section: _SemanticSection,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Keep a feature together; split oversized features at their inner headings."""

    headers = list(section.header_lines)
    body = list(section.body_lines)
    complete = "\n".join([*headers, *body]).strip()
    if len(complete) <= chunk_size:
        return [complete]

    chunks: list[str] = []
    current: list[str] = []
    # Kesilgan bo'lak qaysi ichki bo'limga tegishli ekani yo'qolmasligi uchun
    # oxirgi ichki sarlavha yangi bo'lakning boshiga ko'chiriladi.
    current_subheading = ""

    def flush() -> list[str]:
        nonlocal current
        previous = current
        if current:
            chunks.append("\n".join([*headers, *current]).strip())
        current = []
        return previous

    for block in body:
        is_subheading = block in section.subheadings
        carried = [current_subheading] if current_subheading else []
        header_length = len("\n".join([*headers, *carried]))
        body_limit = max(200, chunk_size - header_length - 1)
        for part in _split_long_paragraph(block, max_length=body_limit):
            candidate = "\n".join([*headers, *current, part]).strip()
            if len(candidate) <= chunk_size:
                current.append(part)
            else:
                previous = flush()
                if is_subheading:
                    current = []
                elif carried:
                    current = list(carried)
                else:
                    # Ichki sarlavha yo'q (jadval qatorlari, tekis matn): oldingi
                    # bo'lakdan qisqa dumini olib qolamiz.
                    current = _overlap_paragraphs(previous, heading="", chunk_overlap=chunk_overlap)
                current.append(part)
            if is_subheading:
                current_subheading = part
                carried = [part]

    if current or not chunks:
        chunks.append("\n".join([*headers, *current]).strip())
    return chunks


# Kept for direct text use and backwards-compatible tests. DOCX indexing uses
# the richer structure-aware path above.
def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(normalized) <= chunk_size:
        return [normalized]
    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    chunks: list[str] = []
    for section in _group_paragraphs_by_section(paragraphs):
        semantic = _SemanticSection(
            title_path=section[0],
            header_lines=(section[0],),
            body_lines=tuple(section[1:]),
        )
        chunks.extend(_split_semantic_section(semantic, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks


def _group_paragraphs_by_section(paragraphs: list[str]) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] = []
    for paragraph in paragraphs:
        if _is_top_level_heading(paragraph) and current:
            sections.append(current)
            current = [paragraph]
        else:
            current.append(paragraph)
    if current:
        sections.append(current)
    return sections


def _is_top_level_heading(paragraph: str) -> bool:
    return bool(re.match(r"^\d+\.\s+\S+", paragraph.strip()))


def _overlap_paragraphs(current: list[str], *, heading: str, chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return [heading] if heading else []
    body = [item for item in current if item and item != heading]
    selected: list[str] = []
    total = 0
    for paragraph in reversed(body):
        next_total = total + len(paragraph)
        if selected and next_total > chunk_overlap:
            break
        selected.insert(0, paragraph)
        total = next_total
    return ([heading] if heading else []) + selected


def _split_long_paragraph(paragraph: str, *, max_length: int) -> list[str]:
    if len(paragraph) <= max_length:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts
