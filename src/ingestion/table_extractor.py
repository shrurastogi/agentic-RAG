"""
Table Extractor for extracting and processing tables from PDFs.

This module extracts tables and converts them to multiple representations:
- Markdown format (for context)
- JSON structure (for filtering/querying)
- LLM-generated summary (for semantic search)

Uses camelot-py for table detection and extraction.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import pandas as pd
from loguru import logger

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot not available - table extraction will be limited")

from .pdf_parser import BoundingBox
from config.settings import settings


@dataclass
class ExtractedTable:
    """Represents an extracted table with multiple representations."""
    table_id: str
    doc_id: str
    page_num: int
    markdown: str
    json_structure: str
    summary: str = ""
    bbox: Optional[BoundingBox] = None
    num_rows: int = 0
    num_cols: int = 0
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "table_id": self.table_id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "markdown": self.markdown,
            "json_structure": self.json_structure,
            "summary": self.summary,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class TableExtractor:
    """
    Extract tables from PDFs using camelot-py.

    Features:
    - Table detection and extraction
    - Multiple output formats (markdown, JSON, summary)
    - Bounding box extraction for citations
    - Confidence scoring
    """

    def __init__(
        self,
        method: str = "lattice",  # "lattice" or "stream"
        generate_summaries: bool = None
    ):
        """
        Initialize table extractor.

        Args:
            method: Extraction method - "lattice" for bordered tables, "stream" for borderless
            generate_summaries: Whether to generate LLM summaries (default from settings)
        """
        if not CAMELOT_AVAILABLE:
            logger.error("Camelot is not installed. Install with: pip install camelot-py[cv]")
            raise ImportError("camelot-py is required for table extraction")

        self.method = method
        self.generate_summaries = (
            generate_summaries
            if generate_summaries is not None
            else settings.GENERATE_TABLE_SUMMARIES
        )

        logger.info(f"TableExtractor initialized: method={method}, summaries={self.generate_summaries}")

    def extract_tables_from_pdf(
        self,
        pdf_path: Path,
        doc_id: Optional[str] = None,
        pages: Optional[str] = None
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a PDF.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier
            pages: Page numbers to process (e.g., "1-3,5", default: all pages)

        Returns:
            List of ExtractedTable objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = doc_id or pdf_path.stem
        logger.info(f"Extracting tables from {pdf_path} using method: {self.method}")

        extracted_tables = []

        try:
            # Extract tables using camelot
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages or 'all',
                flavor=self.method,
                suppress_stdout=True
            )

            logger.info(f"Found {len(tables)} tables in {pdf_path}")

            for idx, table in enumerate(tables):
                try:
                    extracted = self._process_table(table, doc_id, idx)
                    if extracted:
                        extracted_tables.append(extracted)
                except Exception as e:
                    logger.error(f"Error processing table {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")

            # Try alternative method if first one fails
            if self.method == "lattice":
                logger.info("Retrying with 'stream' method...")
                try:
                    tables = camelot.read_pdf(
                        str(pdf_path),
                        pages=pages or 'all',
                        flavor="stream",
                        suppress_stdout=True
                    )

                    for idx, table in enumerate(tables):
                        try:
                            extracted = self._process_table(table, doc_id, idx)
                            if extracted:
                                extracted_tables.append(extracted)
                        except Exception as e:
                            logger.error(f"Error processing table {idx}: {e}")
                            continue

                except Exception as e2:
                    logger.error(f"Both extraction methods failed: {e2}")

        logger.info(f"Successfully extracted {len(extracted_tables)} tables")
        return extracted_tables

    def _process_table(
        self,
        table,
        doc_id: str,
        table_idx: int
    ) -> Optional[ExtractedTable]:
        """
        Process a single camelot table object.

        Args:
            table: Camelot table object
            doc_id: Document ID
            table_idx: Table index

        Returns:
            ExtractedTable or None if processing fails
        """
        try:
            # Get DataFrame
            df = table.df

            # Skip empty tables
            if df.empty or df.shape[0] < 2:
                logger.debug(f"Skipping empty/small table {table_idx}")
                return None

            # Get page number (camelot pages are 1-indexed)
            page_num = table.page - 1

            # Create table ID
            table_id = f"{doc_id}_table_{page_num}_{table_idx}"

            # Convert to different formats
            markdown = self._to_markdown(df)
            json_structure = self._to_json(df)

            # Generate summary if enabled
            summary = ""
            if self.generate_summaries:
                summary = self._generate_summary(df, markdown)

            # Get bounding box
            bbox = self._extract_bbox(table, page_num)

            # Get dimensions
            num_rows, num_cols = df.shape

            # Get confidence score (accuracy from camelot)
            confidence = table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.8

            extracted_table = ExtractedTable(
                table_id=table_id,
                doc_id=doc_id,
                page_num=page_num,
                markdown=markdown,
                json_structure=json_structure,
                summary=summary,
                bbox=bbox,
                num_rows=num_rows,
                num_cols=num_cols,
                confidence=confidence,
                metadata={
                    "extraction_method": self.method,
                    "camelot_accuracy": table.accuracy if hasattr(table, 'accuracy') else None
                }
            )

            logger.debug(f"Processed table {table_id}: {num_rows}x{num_cols}, confidence={confidence:.2f}")

            return extracted_table

        except Exception as e:
            logger.error(f"Error processing table {table_idx}: {e}")
            return None

    def _to_markdown(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to markdown format.

        Args:
            df: Pandas DataFrame

        Returns:
            Markdown string
        """
        try:
            # Clean up the dataframe
            df_clean = df.copy()

            # Use first row as header if it looks like headers
            if self._is_header_row(df_clean.iloc[0]):
                df_clean.columns = df_clean.iloc[0]
                df_clean = df_clean[1:]

            markdown = df_clean.to_markdown(index=False)
            return markdown

        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            return df.to_string()

    def _to_json(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to JSON structure.

        Args:
            df: Pandas DataFrame

        Returns:
            JSON string
        """
        try:
            # Clean up the dataframe
            df_clean = df.copy()

            # Use first row as header if appropriate
            if self._is_header_row(df_clean.iloc[0]):
                df_clean.columns = df_clean.iloc[0]
                df_clean = df_clean[1:]

            # Convert to list of dicts
            json_data = df_clean.to_dict('records')

            return json.dumps(json_data, indent=2)

        except Exception as e:
            logger.error(f"Error converting to JSON: {e}")
            return json.dumps({"data": df.values.tolist()})

    def _is_header_row(self, row: pd.Series) -> bool:
        """
        Check if a row looks like a header row.

        Args:
            row: Pandas Series (row)

        Returns:
            True if row appears to be headers
        """
        # Simple heuristic: if most cells are non-numeric strings
        non_numeric_count = sum(1 for val in row if isinstance(val, str) and not val.replace('.', '').replace('-', '').isdigit())
        return non_numeric_count >= len(row) * 0.7

    def _generate_summary(self, df: pd.DataFrame, markdown: str) -> str:
        """
        Generate a summary of the table.

        For now, creates a simple rule-based summary.
        In Phase 5, this will use LLM for better summaries.

        Args:
            df: Pandas DataFrame
            markdown: Markdown representation

        Returns:
            Summary string
        """
        try:
            num_rows, num_cols = df.shape

            # Get column info
            if self._is_header_row(df.iloc[0]):
                headers = df.iloc[0].tolist()
                summary_parts = [
                    f"Table with {num_rows-1} rows and {num_cols} columns.",
                    f"Columns: {', '.join([str(h) for h in headers[:5]])}{'...' if len(headers) > 5 else ''}."
                ]
            else:
                summary_parts = [f"Table with {num_rows} rows and {num_cols} columns."]

            # Add content preview
            first_few_cells = df.iloc[0, :3].tolist()
            summary_parts.append(f"Sample data: {', '.join([str(c) for c in first_few_cells])}...")

            summary = " ".join(summary_parts)

            # Truncate if too long
            max_length = settings.TABLE_SUMMARY_MAX_LENGTH
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Table with {df.shape[0]} rows and {df.shape[1]} columns"

    def _extract_bbox(self, table, page_num: int) -> Optional[BoundingBox]:
        """
        Extract bounding box from camelot table.

        Args:
            table: Camelot table object
            page_num: Page number (0-indexed)

        Returns:
            BoundingBox or None
        """
        try:
            # Camelot provides bounding box coordinates
            # Format: (x0, y0, x1, y1)
            if hasattr(table, '_bbox'):
                x0, y0, x1, y1 = table._bbox
                return BoundingBox(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page_num=page_num
                )
        except Exception as e:
            logger.debug(f"Could not extract bbox: {e}")

        return None

    def extract_table_from_page(
        self,
        pdf_path: Path,
        page_num: int,
        doc_id: Optional[str] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables from a specific page.

        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            doc_id: Document ID

        Returns:
            List of ExtractedTable objects from that page
        """
        # Camelot uses 1-indexed pages
        camelot_page = page_num + 1

        return self.extract_tables_from_pdf(
            pdf_path,
            doc_id=doc_id,
            pages=str(camelot_page)
        )
