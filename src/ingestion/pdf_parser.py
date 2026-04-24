"""
PDF Parser using PyMuPDF for fast extraction with layout preservation.

This module extracts text, tables, and figures from PDFs while preserving:
- Page numbers
- Bounding boxes (for citation highlighting)
- Content regions (text blocks, tables, figures)
- Section structure
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class BoundingBox:
    """Represents a rectangular region in a PDF page."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_num: int

    def to_dict(self) -> Dict:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page_num": self.page_num
        }


@dataclass
class ContentRegion:
    """Represents a content region in a PDF (text, table, or figure)."""
    content_type: str  # "text", "table", "figure"
    bbox: BoundingBox
    page_num: int
    content: str = ""
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class TextBlock:
    """Represents a text block with metadata."""
    text: str
    page_num: int
    bbox: BoundingBox
    font_size: float = 0.0
    font_name: str = ""
    is_header: bool = False
    section_title: Optional[str] = None


@dataclass
class ParsedDocument:
    """Container for parsed PDF content."""
    doc_id: str
    title: str
    total_pages: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    content_regions: List[ContentRegion] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class PDFParser:
    """
    PDF parser using PyMuPDF for fast, layout-aware extraction.

    Features:
    - Fast text extraction with bounding boxes
    - Content region detection (text, tables, figures)
    - Section hierarchy preservation
    - Font analysis for header detection
    """

    def __init__(self, preserve_layout: bool = True):
        """
        Initialize PDF parser.

        Args:
            preserve_layout: Whether to preserve document layout structure
        """
        self.preserve_layout = preserve_layout

    def parse_pdf(self, pdf_path: Path, doc_id: Optional[str] = None) -> ParsedDocument:
        """
        Parse a PDF file and extract all content with metadata.

        Args:
            pdf_path: Path to the PDF file
            doc_id: Optional document ID (defaults to filename)

        Returns:
            ParsedDocument containing all extracted content
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_id = doc_id or pdf_path.stem
        logger.info(f"Parsing PDF: {pdf_path} (doc_id: {doc_id})")

        try:
            doc = fitz.open(pdf_path)
            parsed_doc = ParsedDocument(
                doc_id=doc_id,
                title=self._extract_title(doc),
                total_pages=len(doc),
                metadata=self._extract_metadata(doc)
            )

            # Parse each page
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text blocks with bounding boxes
                text_blocks = self._extract_text_blocks(page, page_num)
                parsed_doc.text_blocks.extend(text_blocks)

                # Detect content regions (tables, figures)
                content_regions = self._detect_content_regions(page, page_num)
                parsed_doc.content_regions.extend(content_regions)

            doc.close()

            # Analyze section structure
            self._identify_sections(parsed_doc)

            logger.info(
                f"Parsed {parsed_doc.total_pages} pages: "
                f"{len(parsed_doc.text_blocks)} text blocks, "
                f"{len(parsed_doc.content_regions)} content regions"
            )

            return parsed_doc

        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {e}")
            raise

    def _extract_title(self, doc: fitz.Document) -> str:
        """Extract document title from metadata or first page."""
        # Try metadata first
        title = doc.metadata.get("title", "").strip()
        if title:
            return title

        # Fallback: use first large text on first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]

            # Find largest text block (likely title)
            max_size = 0
            title_text = ""
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span["size"] > max_size:
                                max_size = span["size"]
                                title_text = span["text"].strip()

            if title_text:
                return title_text

        return "Untitled"

    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """Extract document metadata."""
        metadata = {}

        if doc.metadata:
            metadata.update({
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", "")
            })

        return metadata

    def _extract_text_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """
        Extract text blocks from a page with bounding boxes and font info.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            List of TextBlock objects
        """
        text_blocks = []
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    # Extract text and font info from spans
                    line_text = ""
                    font_sizes = []
                    font_names = []

                    for span in line.get("spans", []):
                        line_text += span["text"]
                        font_sizes.append(span["size"])
                        font_names.append(span["font"])

                    if not line_text.strip():
                        continue

                    # Create bounding box
                    bbox_coords = line["bbox"]
                    bbox = BoundingBox(
                        x0=bbox_coords[0],
                        y0=bbox_coords[1],
                        x1=bbox_coords[2],
                        y1=bbox_coords[3],
                        page_num=page_num
                    )

                    # Average font size
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                    font_name = font_names[0] if font_names else ""

                    # Detect if this is likely a header (larger font)
                    is_header = avg_font_size > 14  # Heuristic threshold

                    text_block = TextBlock(
                        text=line_text.strip(),
                        page_num=page_num,
                        bbox=bbox,
                        font_size=avg_font_size,
                        font_name=font_name,
                        is_header=is_header
                    )

                    text_blocks.append(text_block)

        return text_blocks

    def _detect_content_regions(self, page: fitz.Page, page_num: int) -> List[ContentRegion]:
        """
        Detect content regions (tables and figures) using heuristics.

        This is a basic implementation - Phase 2 will add specialized extractors.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)

        Returns:
            List of ContentRegion objects
        """
        content_regions = []

        # Detect images (potential figures)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            # Get image bounding box
            img_bbox = page.get_image_bbox(img[7])  # img[7] is the xref

            bbox = BoundingBox(
                x0=img_bbox.x0,
                y0=img_bbox.y0,
                x1=img_bbox.x1,
                y1=img_bbox.y1,
                page_num=page_num
            )

            region = ContentRegion(
                content_type="figure",
                bbox=bbox,
                page_num=page_num,
                metadata={"image_index": img_index, "xref": img[7]}
            )
            content_regions.append(region)

        # Detect tables using simple heuristics (will be improved in Phase 2)
        # Tables often have: aligned text, grid-like structure, multiple columns
        blocks = page.get_text("dict")["blocks"]

        # Simple heuristic: look for blocks with multiple lines in grid pattern
        for block in blocks:
            if block.get("type") == 0:  # Text block
                lines = block.get("lines", [])

                # If block has many lines with similar x-coordinates (columnar)
                if len(lines) > 5:  # At least 5 rows
                    x_coords = [line["bbox"][0] for line in lines]

                    # Check if x-coordinates are aligned (variance is low)
                    if len(set([round(x, -1) for x in x_coords])) > 1:
                        bbox_coords = block["bbox"]
                        bbox = BoundingBox(
                            x0=bbox_coords[0],
                            y0=bbox_coords[1],
                            x1=bbox_coords[2],
                            y1=bbox_coords[3],
                            page_num=page_num
                        )

                        region = ContentRegion(
                            content_type="table",
                            bbox=bbox,
                            page_num=page_num,
                            confidence=0.5  # Low confidence - needs Phase 2 improvement
                        )
                        content_regions.append(region)

        return content_regions

    def _identify_sections(self, parsed_doc: ParsedDocument) -> None:
        """
        Identify section hierarchy based on font sizes and formatting.

        Updates text_blocks in-place with section_title metadata.

        Args:
            parsed_doc: ParsedDocument to analyze
        """
        current_section = None

        for block in parsed_doc.text_blocks:
            if block.is_header:
                # This is a section header
                current_section = block.text
            else:
                # Assign current section to this block
                block.section_title = current_section

    def extract_page_text(self, pdf_path: Path, page_num: int) -> str:
        """
        Extract raw text from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            Text content of the page
        """
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            raise ValueError(f"Page {page_num} out of range (total: {len(doc)})")

        page = doc[page_num]
        text = page.get_text()
        doc.close()

        return text

    def get_text_with_bbox(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: BoundingBox
    ) -> str:
        """
        Extract text from a specific region (for citation verification).

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            bbox: Bounding box defining the region

        Returns:
            Text content within the bounding box
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Create rectangle from bounding box
        rect = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)

        # Extract text from region
        text = page.get_text("text", clip=rect)
        doc.close()

        return text.strip()
