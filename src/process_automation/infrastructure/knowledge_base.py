"""Loading and parsing of the company's SOP corpus.

The knowledge base is a folder of Markdown documents with a YAML-style front
matter block and a table of approved steps. Keeping the corpus in plain files
means a new process is on-boarded by *dropping a document in the folder* - no
Python change at all (Open/Closed Principle at the data level).

SOLID notes
-----------
* SRP - :class:`SopDocument` holds one parsed document, :class:`KnowledgeBase`
        holds the collection, :class:`MarkdownKnowledgeBaseLoader` does the
        file I/O. Splitting the three keeps each one trivially testable.
* DIP - the RAG agent never imports this module; it receives a
        :class:`~process_automation.interfaces.DocumentRetriever` instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from ..domain.models import AutomationClass, ProcessStep, StandardProcess

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
#: A Markdown table row: ``| 2 | Direct manager approval | ... |``
_TABLE_ROW_RE = re.compile(r"^\|(?!\s*[-: ]+\|)(.+)\|\s*$", re.MULTILINE)
_CONTROL_REF_RE = re.compile(r"\(([^)]+)\)")


@dataclass(frozen=True)
class TextChunk:
    """One retrievable unit of a document."""

    chunk_id: str
    document_id: str
    section: str
    text: str


@dataclass(frozen=True)
class SopDocument:
    """A parsed standard-operating-procedure document."""

    document_id: str
    process: str
    sla_hours: float
    owner: str
    aliases: tuple[str, ...] = ()
    steps: tuple[ProcessStep, ...] = ()
    policy_notes: tuple[str, ...] = ()
    chunks: tuple[TextChunk, ...] = ()
    source_path: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_standard_process(self) -> StandardProcess:
        """Project the document onto the domain's *should-be* value object."""
        return StandardProcess(
            process_name=self.process,
            sla_hours=self.sla_hours,
            steps=self.steps,
            policy_notes=self.policy_notes,
            source_documents=(f"{self.document_id} ({Path(self.source_path).name})",),
        )

    @property
    def searchable_header(self) -> str:
        """Synthetic chunk that makes the document findable by name/alias."""
        aliases = ", ".join(self.aliases)
        return (
            f"{self.document_id} {self.process}. "
            f"Also known as: {aliases}. "
            f"Owner: {self.owner}. Documented SLA: {self.sla_hours:g} hours."
        )


class KnowledgeBase:
    """In-memory collection of :class:`SopDocument` objects."""

    def __init__(self, documents: Sequence[SopDocument] = ()) -> None:
        self._documents: dict[str, SopDocument] = {
            document.document_id: document for document in documents
        }

    def add(self, document: SopDocument) -> "KnowledgeBase":
        self._documents[document.document_id] = document
        return self

    def get(self, document_id: str) -> SopDocument | None:
        return self._documents.get(document_id)

    @property
    def documents(self) -> tuple[SopDocument, ...]:
        return tuple(self._documents.values())

    def all_chunks(self) -> tuple[TextChunk, ...]:
        """Every chunk of every document, ready to be indexed."""
        return tuple(chunk for document in self._documents.values() for chunk in document.chunks)

    def __len__(self) -> int:
        return len(self._documents)


class MarkdownKnowledgeBaseLoader:
    """Reads ``*.md`` SOP files and turns them into documents + chunks."""

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    def load(self) -> KnowledgeBase:
        """Parse every Markdown file in the configured directory."""
        if not self._directory.is_dir():
            raise FileNotFoundError(f"knowledge base directory not found: {self._directory}")
        documents = [
            self._parse_document(path)
            for path in sorted(self._directory.glob("*.md"))
        ]
        if not documents:
            raise ValueError(f"no SOP documents found in {self._directory}")
        return KnowledgeBase(documents)

    # -- parsing helpers ---------------------------------------------------

    def _parse_document(self, path: Path) -> SopDocument:
        raw = path.read_text(encoding="utf-8")
        front_matter, body = self._split_front_matter(raw)

        document_id = front_matter.get("document_id", path.stem)
        steps = self._parse_steps(body)
        document = SopDocument(
            document_id=document_id,
            process=front_matter.get("process", path.stem),
            sla_hours=float(front_matter.get("sla_hours", 0) or 0),
            owner=front_matter.get("owner", "unknown"),
            aliases=tuple(
                alias.strip()
                for alias in front_matter.get("aliases", "").split(",")
                if alias.strip()
            ),
            steps=steps,
            policy_notes=self._parse_policy_notes(body),
            source_path=str(path),
            metadata=front_matter,
        )
        # Chunks are built last because the header chunk needs the parsed
        # metadata (process name, aliases, SLA) to be searchable.
        return replace(document, chunks=self._build_chunks(document, body))

    @staticmethod
    def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
        """Return the ``key: value`` front matter and the remaining body."""
        match = _FRONT_MATTER_RE.match(raw)
        if not match:
            return {}, raw
        front_matter: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            front_matter[key.strip()] = value.strip()
        return front_matter, raw[match.end():]

    @staticmethod
    def _parse_steps(body: str) -> tuple[ProcessStep, ...]:
        """Read the 'Approved steps' Markdown table into domain objects."""
        steps: list[ProcessStep] = []
        for row in _TABLE_ROW_RE.findall(body):
            cells = [cell.strip() for cell in row.split("|")]
            if len(cells) < 6 or not cells[0].isdigit():
                continue  # header row, or a table that is not the step table
            order, title, owner, target, kind, control = cells[:6]
            control_flag = control.lower().startswith("yes")
            reference = _CONTROL_REF_RE.search(control)
            steps.append(
                ProcessStep(
                    order=int(order),
                    title=title,
                    owner=owner,
                    automation_class=_parse_automation_class(kind),
                    target_hours=_parse_float(target),
                    is_mandatory_control=control_flag,
                    control_reference=reference.group(1) if reference else None,
                )
            )
        return tuple(sorted(steps, key=lambda step: step.order))

    @staticmethod
    def _parse_policy_notes(body: str) -> tuple[str, ...]:
        """Collect the bullet list of the 'Policy notes' section."""
        sections = _split_sections(body)
        for title, content in sections:
            if "policy" not in title.lower():
                continue
            notes: list[str] = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- "):
                    notes.append(stripped[2:].strip())
                elif notes and stripped:
                    # Continuation line of the previous bullet.
                    notes[-1] = f"{notes[-1]} {stripped}"
            return tuple(notes)
        return ()

    @staticmethod
    def _build_chunks(document: SopDocument, body: str) -> tuple[TextChunk, ...]:
        """One chunk per '##' section, plus a synthetic header chunk.

        Section-level chunking fits SOP documents far better than a blind
        character window: each section is already a self-contained idea, so a
        retrieved chunk never cuts a step table in half.
        """
        chunks = [
            TextChunk(
                chunk_id=f"{document.document_id}#header",
                document_id=document.document_id,
                section="header",
                text=document.searchable_header,
            )
        ]
        for index, (title, content) in enumerate(_split_sections(body), start=1):
            text = f"{title}\n{content}".strip()
            if not text:
                continue
            chunks.append(
                TextChunk(
                    chunk_id=f"{document.document_id}#{index}",
                    document_id=document.document_id,
                    section=title,
                    # Prefixing the process name keeps every chunk attributable
                    # even when it is read in isolation by the parser or an LLM.
                    text=f"[{document.process}] {text}",
                )
            )
        return tuple(chunks)


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a Markdown body on its ``##`` headings."""
    matches = list(_SECTION_RE.finditer(body))
    if not matches:
        return [("document", body.strip())]
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        sections.append((match.group(1).strip(), body[start:end].strip()))
    return sections


def _parse_automation_class(value: str) -> AutomationClass:
    """Map the table's ``Type`` column onto the enum, defaulting safely."""
    normalised = value.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return AutomationClass(normalised)
    except ValueError:
        # Unknown vocabulary must not crash the load: treat it as manual work,
        # which is the conservative assumption for a process analysis.
        return AutomationClass.DATA_ENTRY


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "KnowledgeBase",
    "MarkdownKnowledgeBaseLoader",
    "SopDocument",
    "TextChunk",
]
