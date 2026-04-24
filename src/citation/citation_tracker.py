"""
Citation Tracker for managing citations in RAG responses.

Features:
- Assign unique citation IDs
- Store citation database
- Expand inline citations in text
- Track provenance to source documents
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class Citation:
    """Represents a single citation."""
    citation_id: str
    doc_id: str
    title: str
    page_num: int
    section: Optional[str]
    snippet: str
    content_type: str  # "text", "table", "figure"
    score: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "citation_id": self.citation_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "page_num": self.page_num,
            "section": self.section,
            "snippet": self.snippet,
            "content_type": self.content_type,
            "score": self.score,
            "metadata": self.metadata
        }

    def format_reference(self) -> str:
        """Format as a reference string."""
        ref = f"{self.citation_id}: {self.doc_id}"

        if self.title and self.title != self.doc_id:
            ref += f" - {self.title}"

        ref += f", Page {self.page_num}"

        if self.section:
            ref += f", Section: {self.section}"

        return ref


class CitationTracker:
    """
    Track and manage citations in RAG responses.

    Provides:
    - Unique citation ID assignment
    - Citation database storage
    - Inline citation expansion
    """

    def __init__(self):
        """Initialize citation tracker."""
        self.citations: Dict[str, Citation] = {}
        self.citation_counter = 0

        logger.info("CitationTracker initialized")

    def add_citation(
        self,
        doc_id: str,
        title: str,
        page_num: int,
        snippet: str,
        section: Optional[str] = None,
        content_type: str = "text",
        score: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a new citation and return its ID.

        Args:
            doc_id: Document identifier
            title: Document title
            page_num: Page number
            snippet: Text snippet
            section: Section title
            content_type: Type of content
            score: Relevance score
            metadata: Additional metadata

        Returns:
            Citation ID (e.g., "CIT-001")
        """
        self.citation_counter += 1
        citation_id = f"CIT-{self.citation_counter:03d}"

        citation = Citation(
            citation_id=citation_id,
            doc_id=doc_id,
            title=title,
            page_num=page_num,
            section=section,
            snippet=snippet[:200],  # Truncate snippet
            content_type=content_type,
            score=score,
            metadata=metadata or {}
        )

        self.citations[citation_id] = citation

        logger.debug(f"Added citation: {citation_id} -> {doc_id}, p.{page_num}")

        return citation_id

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """
        Get citation by ID.

        Args:
            citation_id: Citation ID

        Returns:
            Citation or None
        """
        return self.citations.get(citation_id)

    def get_all_citations(self) -> List[Citation]:
        """
        Get all citations.

        Returns:
            List of all citations
        """
        return list(self.citations.values())

    def format_references(self) -> str:
        """
        Format all citations as a reference list.

        Returns:
            Formatted reference string
        """
        if not self.citations:
            return "No citations."

        refs = ["## References\n"]

        for citation in sorted(
            self.citations.values(),
            key=lambda c: c.citation_id
        ):
            refs.append(f"{citation.format_reference()}")

        return "\n".join(refs)

    def expand_inline_citations(self, text: str) -> str:
        """
        Expand inline citation markers in text.

        Converts [CIT-001] to full references.

        Args:
            text: Text with citation markers

        Returns:
            Text with expanded citations
        """
        import re

        # Find all citation markers
        markers = re.findall(r'\[CIT-\d{3}\]', text)

        for marker in set(markers):
            citation_id = marker.strip('[]')
            citation = self.get_citation(citation_id)

            if citation:
                # Replace marker with expanded reference
                expansion = f" [{citation.doc_id}, p.{citation.page_num}]"
                text = text.replace(marker, expansion)

        return text

    def reset(self):
        """Reset citation tracker."""
        self.citations = {}
        self.citation_counter = 0
        logger.info("Citation tracker reset")
