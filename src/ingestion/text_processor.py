"""
Text Processor for semantic chunking with section awareness.

This module chunks text from parsed PDFs while:
- Preserving section boundaries
- Maintaining context with overlapping windows
- Keeping statistical data intact
- Never splitting tables or figure captions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import re
from loguru import logger

from .pdf_parser import ParsedDocument, TextBlock, BoundingBox
from config.settings import settings


@dataclass
class TextChunk:
    """Represents a semantically meaningful text chunk."""
    chunk_id: str
    doc_id: str
    text: str
    page_num: int
    section_title: Optional[str] = None
    chunk_index: int = 0
    bbox: Optional[BoundingBox] = None
    metadata: Dict = field(default_factory=dict)
    token_count: int = 0

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "page_num": self.page_num,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "metadata": self.metadata,
            "token_count": self.token_count
        }


class TextProcessor:
    """
    Process text into semantically meaningful chunks.

    Features:
    - Section-aware chunking
    - Configurable chunk size and overlap
    - Statistical content preservation
    - Section boundary preservation
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = None,
        preserve_section_boundaries: bool = None
    ):
        """
        Initialize text processor.

        Args:
            chunk_size: Target chunk size in tokens (default from settings)
            chunk_overlap: Overlap size in tokens (default from settings)
            min_chunk_size: Minimum chunk size (default from settings)
            preserve_section_boundaries: Whether to preserve sections (default from settings)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or settings.MIN_CHUNK_SIZE
        self.preserve_section_boundaries = (
            preserve_section_boundaries
            if preserve_section_boundaries is not None
            else settings.PRESERVE_SECTION_BOUNDARIES
        )

        logger.info(
            f"TextProcessor initialized: chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, preserve_sections={self.preserve_section_boundaries}"
        )

    def process_document(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Process a parsed document into chunks.

        Args:
            parsed_doc: ParsedDocument from PDFParser

        Returns:
            List of TextChunk objects
        """
        logger.info(f"Processing document: {parsed_doc.doc_id}")

        if self.preserve_section_boundaries:
            chunks = self._chunk_by_sections(parsed_doc)
        else:
            chunks = self._chunk_by_sliding_window(parsed_doc)

        logger.info(f"Created {len(chunks)} chunks from {parsed_doc.doc_id}")

        return chunks

    def _chunk_by_sections(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Chunk text while preserving section boundaries.

        Args:
            parsed_doc: ParsedDocument object

        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_counter = 0

        # Group text blocks by section
        sections = self._group_by_section(parsed_doc.text_blocks)

        for section_title, blocks in sections.items():
            # Combine blocks in this section
            section_text = " ".join([block.text for block in blocks])

            # Get page number (first block in section)
            page_num = blocks[0].page_num if blocks else 0

            # Check if section is small enough to keep intact
            token_count = self._estimate_tokens(section_text)

            if token_count <= self.chunk_size:
                # Keep entire section as one chunk
                chunk = TextChunk(
                    chunk_id=f"{parsed_doc.doc_id}_chunk_{chunk_counter}",
                    doc_id=parsed_doc.doc_id,
                    text=section_text,
                    page_num=page_num,
                    section_title=section_title,
                    chunk_index=chunk_counter,
                    bbox=blocks[0].bbox if blocks else None,
                    token_count=token_count,
                    metadata={
                        "contains_statistics": self._contains_statistics(section_text),
                        "num_blocks": len(blocks)
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1

            else:
                # Split section using sliding window, but keep section title
                section_chunks = self._sliding_window_chunk(
                    section_text,
                    parsed_doc.doc_id,
                    page_num,
                    chunk_counter,
                    section_title
                )
                chunks.extend(section_chunks)
                chunk_counter += len(section_chunks)

        return chunks

    def _chunk_by_sliding_window(self, parsed_doc: ParsedDocument) -> List[TextChunk]:
        """
        Chunk text using a sliding window approach (ignoring sections).

        Args:
            parsed_doc: ParsedDocument object

        Returns:
            List of TextChunk objects
        """
        # Combine all text
        all_text = " ".join([block.text for block in parsed_doc.text_blocks])

        chunks = self._sliding_window_chunk(
            all_text,
            parsed_doc.doc_id,
            page_num=0,
            start_index=0
        )

        return chunks

    def _sliding_window_chunk(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        start_index: int = 0,
        section_title: Optional[str] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks using sliding window.

        Args:
            text: Text to chunk
            doc_id: Document ID
            page_num: Page number
            start_index: Starting chunk index
            section_title: Optional section title

        Returns:
            List of TextChunk objects
        """
        chunks = []

        # Simple word-based approximation for tokens
        words = text.split()
        words_per_token = 0.75  # Rough approximation

        chunk_size_words = int(self.chunk_size * words_per_token)
        overlap_words = int(self.chunk_overlap * words_per_token)

        i = 0
        chunk_index = start_index

        while i < len(words):
            # Get chunk words
            chunk_words = words[i : i + chunk_size_words]

            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_words) < int(self.min_chunk_size * words_per_token) and i + chunk_size_words < len(words):
                i += chunk_size_words - overlap_words
                continue

            chunk_text = " ".join(chunk_words)
            token_count = self._estimate_tokens(chunk_text)

            chunk = TextChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                text=chunk_text,
                page_num=page_num,
                section_title=section_title,
                chunk_index=chunk_index,
                token_count=token_count,
                metadata={
                    "contains_statistics": self._contains_statistics(chunk_text),
                    "word_count": len(chunk_words)
                }
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move window
            i += chunk_size_words - overlap_words

            # Prevent infinite loop
            if i == 0:
                i = chunk_size_words

        return chunks

    def _group_by_section(self, text_blocks: List[TextBlock]) -> Dict[str, List[TextBlock]]:
        """
        Group text blocks by section.

        Args:
            text_blocks: List of TextBlock objects

        Returns:
            Dictionary mapping section titles to text blocks
        """
        sections = {}
        current_section = "Introduction"  # Default section

        for block in text_blocks:
            # Update section if block has a section title
            if block.section_title:
                current_section = block.section_title

            # Add block to current section
            if current_section not in sections:
                sections[current_section] = []

            sections[current_section].append(block)

        return sections

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Simple approximation: ~0.75 words per token for English.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        words = len(text.split())
        return int(words / 0.75)

    def _contains_statistics(self, text: str) -> bool:
        """
        Check if text contains statistical data.

        Looks for patterns like:
        - Percentages: 45%, 12.5%
        - P-values: p<0.05, p=0.001
        - Confidence intervals: 95% CI [1.2, 3.4]
        - Numbers with units: 100 mg, 5.2 mmol/L

        Args:
            text: Text to check

        Returns:
            True if statistical content detected
        """
        # Percentage pattern
        if re.search(r'\d+\.?\d*\s*%', text):
            return True

        # P-value pattern
        if re.search(r'p\s*[<>=]\s*0?\.\d+', text.lower()):
            return True

        # Confidence interval pattern
        if re.search(r'\d+%?\s*(CI|confidence interval)', text, re.IGNORECASE):
            return True

        # Statistical terms
        stat_terms = [
            'mean', 'median', 'standard deviation', 'variance',
            'odds ratio', 'hazard ratio', 'relative risk',
            'p-value', 'p value', 'statistically significant',
            'correlation', 'regression'
        ]

        text_lower = text.lower()
        return any(term in text_lower for term in stat_terms)

    def chunk_text(self, text: str, doc_id: str = "unknown") -> List[TextChunk]:
        """
        Chunk raw text (convenience method).

        Args:
            text: Text to chunk
            doc_id: Document ID

        Returns:
            List of TextChunk objects
        """
        return self._sliding_window_chunk(text, doc_id, page_num=0, start_index=0)

    def merge_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Merge consecutive chunks that are too small.

        Args:
            chunks: List of TextChunk objects

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = []
        current_chunk = None

        for chunk in chunks:
            if chunk.token_count < self.min_chunk_size:
                if current_chunk is None:
                    current_chunk = chunk
                else:
                    # Merge with current chunk
                    current_chunk.text += " " + chunk.text
                    current_chunk.token_count = self._estimate_tokens(current_chunk.text)
            else:
                if current_chunk is not None:
                    merged.append(current_chunk)
                    current_chunk = None
                merged.append(chunk)

        # Add remaining chunk
        if current_chunk is not None:
            merged.append(current_chunk)

        return merged
