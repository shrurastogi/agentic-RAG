"""
Document Loader - Orchestrates the complete multi-modal ingestion pipeline.

This module coordinates:
- PDF parsing
- Text chunking
- Table extraction
- Figure processing
- Parallel processing of different content types
- Error handling and progress tracking
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
import json
from datetime import datetime

from .pdf_parser import PDFParser, ParsedDocument
from .text_processor import TextProcessor, TextChunk
from .table_extractor import TableExtractor, ExtractedTable
from .figure_processor import FigureProcessor, ExtractedFigure
from config.settings import settings


@dataclass
class ProcessedDocument:
    """Container for a fully processed document."""
    doc_id: str
    title: str
    pdf_path: str
    total_pages: int
    text_chunks: List[TextChunk] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    figures: List[ExtractedFigure] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "num_chunks": len(self.text_chunks),
            "num_tables": len(self.tables),
            "num_figures": len(self.figures),
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "errors": self.errors
        }

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            "Total Chunks": len(self.text_chunks),
            "Total Tables": len(self.tables),
            "Total Figures": len(self.figures),
            "Total Pages": self.total_pages,
            "Processing Time": f"{self.processing_time:.2f}s",
            "Errors": len(self.errors)
        }


class DocumentLoader:
    """
    Orchestrate complete multi-modal document processing pipeline.

    Features:
    - End-to-end PDF processing
    - Parallel extraction of tables and figures
    - Error handling with graceful degradation
    - Progress tracking
    - Artifact caching
    """

    def __init__(
        self,
        enable_tables: bool = True,
        enable_figures: bool = True,
        enable_caching: bool = True,
        max_workers: int = None
    ):
        """
        Initialize document loader.

        Args:
            enable_tables: Whether to extract tables
            enable_figures: Whether to extract figures
            enable_caching: Whether to cache processed artifacts
            max_workers: Max parallel workers (default from settings)
        """
        self.enable_tables = enable_tables
        self.enable_figures = enable_figures
        self.enable_caching = enable_caching
        self.max_workers = max_workers or settings.MAX_WORKERS

        # Initialize processors
        self.pdf_parser = PDFParser()
        self.text_processor = TextProcessor()

        if self.enable_tables:
            try:
                self.table_extractor = TableExtractor()
            except ImportError:
                logger.warning("Table extraction disabled - camelot not available")
                self.enable_tables = False

        if self.enable_figures:
            self.figure_processor = FigureProcessor()

        # Cache directory
        self.cache_dir = settings.PROCESSED_DATA_DIR / "cache"
        if self.enable_caching:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"DocumentLoader initialized: "
            f"tables={self.enable_tables}, figures={self.enable_figures}, "
            f"caching={self.enable_caching}, workers={self.max_workers}"
        )

    def load_document(
        self,
        pdf_path: Path,
        doc_id: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process a PDF document end-to-end.

        Args:
            pdf_path: Path to PDF file
            doc_id: Optional document ID

        Returns:
            ProcessedDocument with all extracted content
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = doc_id or pdf_path.stem
        start_time = datetime.now()

        logger.info(f"Loading document: {pdf_path} (doc_id: {doc_id})")

        # Check cache
        if self.enable_caching:
            cached = self._load_from_cache(doc_id)
            if cached:
                logger.info(f"Loaded {doc_id} from cache")
                return cached

        errors = []

        # Step 1: Parse PDF
        logger.info("[1/4] Parsing PDF...")
        try:
            parsed_doc = self.pdf_parser.parse_pdf(pdf_path, doc_id=doc_id)
            logger.info(
                f"✓ Parsed {parsed_doc.total_pages} pages, "
                f"{len(parsed_doc.text_blocks)} text blocks"
            )
        except Exception as e:
            logger.error(f"PDF parsing failed: {e}")
            raise

        # Step 2: Process text into chunks
        logger.info("[2/4] Processing text chunks...")
        try:
            text_chunks = self.text_processor.process_document(parsed_doc)
            logger.info(f"✓ Created {len(text_chunks)} chunks")
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            errors.append(f"Text processing error: {e}")
            text_chunks = []

        # Step 3 & 4: Extract tables and figures in parallel
        logger.info("[3/4] Extracting tables and figures...")
        tables, figures = self._extract_multimodal_content(pdf_path, doc_id, errors)

        # Create processed document
        processing_time = (datetime.now() - start_time).total_seconds()

        processed_doc = ProcessedDocument(
            doc_id=doc_id,
            title=parsed_doc.title,
            pdf_path=str(pdf_path),
            total_pages=parsed_doc.total_pages,
            text_chunks=text_chunks,
            tables=tables,
            figures=figures,
            metadata={
                **parsed_doc.metadata,
                "processed_at": datetime.now().isoformat(),
                "num_text_blocks": len(parsed_doc.text_blocks),
                "num_content_regions": len(parsed_doc.content_regions)
            },
            processing_time=processing_time,
            errors=errors
        )

        logger.info(
            f"✓ Document processed in {processing_time:.2f}s: "
            f"{len(text_chunks)} chunks, {len(tables)} tables, {len(figures)} figures"
        )

        # Cache results
        if self.enable_caching:
            self._save_to_cache(processed_doc)

        return processed_doc

    def _extract_multimodal_content(
        self,
        pdf_path: Path,
        doc_id: str,
        errors: List[str]
    ) -> Tuple[List[ExtractedTable], List[ExtractedFigure]]:
        """
        Extract tables and figures in parallel.

        Args:
            pdf_path: Path to PDF
            doc_id: Document ID
            errors: List to append errors to

        Returns:
            Tuple of (tables, figures)
        """
        tables = []
        figures = []

        tasks = []

        # Submit tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            if self.enable_tables:
                tasks.append(
                    executor.submit(
                        self._extract_tables_safe,
                        pdf_path,
                        doc_id,
                        errors
                    )
                )

            if self.enable_figures:
                tasks.append(
                    executor.submit(
                        self._extract_figures_safe,
                        pdf_path,
                        doc_id,
                        errors
                    )
                )

            # Collect results
            for future in as_completed(tasks):
                try:
                    result_type, result_data = future.result()

                    if result_type == "tables":
                        tables = result_data
                        logger.info(f"✓ Extracted {len(tables)} tables")

                    elif result_type == "figures":
                        figures = result_data
                        logger.info(f"✓ Extracted {len(figures)} figures")

                except Exception as e:
                    logger.error(f"Parallel extraction error: {e}")
                    errors.append(f"Extraction error: {e}")

        return tables, figures

    def _extract_tables_safe(
        self,
        pdf_path: Path,
        doc_id: str,
        errors: List[str]
    ) -> Tuple[str, List[ExtractedTable]]:
        """
        Extract tables with error handling.

        Args:
            pdf_path: Path to PDF
            doc_id: Document ID
            errors: Error list

        Returns:
            Tuple of ("tables", list of tables)
        """
        try:
            tables = self.table_extractor.extract_tables_from_pdf(pdf_path, doc_id)
            return ("tables", tables)
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            errors.append(f"Table extraction error: {e}")
            return ("tables", [])

    def _extract_figures_safe(
        self,
        pdf_path: Path,
        doc_id: str,
        errors: List[str]
    ) -> Tuple[str, List[ExtractedFigure]]:
        """
        Extract figures with error handling.

        Args:
            pdf_path: Path to PDF
            doc_id: Document ID
            errors: Error list

        Returns:
            Tuple of ("figures", list of figures)
        """
        try:
            figures = self.figure_processor.extract_figures_from_pdf(pdf_path, doc_id)
            return ("figures", figures)
        except Exception as e:
            logger.error(f"Figure extraction failed: {e}")
            errors.append(f"Figure extraction error: {e}")
            return ("figures", [])

    def _get_cache_path(self, doc_id: str) -> Path:
        """Get cache file path for a document."""
        return self.cache_dir / f"{doc_id}_processed.json"

    def _save_to_cache(self, processed_doc: ProcessedDocument):
        """Save processed document to cache."""
        try:
            cache_path = self._get_cache_path(processed_doc.doc_id)

            cache_data = {
                "doc_id": processed_doc.doc_id,
                "title": processed_doc.title,
                "pdf_path": processed_doc.pdf_path,
                "total_pages": processed_doc.total_pages,
                "text_chunks": [chunk.to_dict() for chunk in processed_doc.text_chunks],
                "tables": [table.to_dict() for table in processed_doc.tables],
                "figures": [figure.to_dict() for figure in processed_doc.figures],
                "metadata": processed_doc.metadata,
                "processing_time": processed_doc.processing_time,
                "errors": processed_doc.errors
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

            logger.debug(f"Saved to cache: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self, doc_id: str) -> Optional[ProcessedDocument]:
        """Load processed document from cache."""
        try:
            cache_path = self._get_cache_path(doc_id)

            if not cache_path.exists():
                return None

            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Reconstruct text chunks
            text_chunks = [
                TextChunk(**chunk_data)
                for chunk_data in cache_data.get("text_chunks", [])
            ]

            # Reconstruct tables
            tables = [
                ExtractedTable(**table_data)
                for table_data in cache_data.get("tables", [])
            ]

            # Reconstruct figures
            figures = [
                ExtractedFigure(**figure_data)
                for figure_data in cache_data.get("figures", [])
            ]

            processed_doc = ProcessedDocument(
                doc_id=cache_data["doc_id"],
                title=cache_data["title"],
                pdf_path=cache_data["pdf_path"],
                total_pages=cache_data["total_pages"],
                text_chunks=text_chunks,
                tables=tables,
                figures=figures,
                metadata=cache_data.get("metadata", {}),
                processing_time=cache_data.get("processing_time", 0.0),
                errors=cache_data.get("errors", [])
            )

            return processed_doc

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def batch_load_documents(
        self,
        pdf_paths: List[Path],
        show_progress: bool = True
    ) -> List[ProcessedDocument]:
        """
        Process multiple PDFs.

        Args:
            pdf_paths: List of PDF paths
            show_progress: Whether to show progress

        Returns:
            List of ProcessedDocument objects
        """
        logger.info(f"Processing {len(pdf_paths)} documents...")

        processed_docs = []

        for i, pdf_path in enumerate(pdf_paths, 1):
            if show_progress:
                logger.info(f"[{i}/{len(pdf_paths)}] Processing {pdf_path.name}...")

            try:
                processed_doc = self.load_document(pdf_path)
                processed_docs.append(processed_doc)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        logger.info(
            f"✓ Batch processing complete: "
            f"{len(processed_docs)}/{len(pdf_paths)} documents processed"
        )

        return processed_docs
