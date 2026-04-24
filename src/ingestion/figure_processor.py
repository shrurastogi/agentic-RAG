"""
Figure Processor for extracting and processing figures/images from PDFs.

This module:
- Extracts figures as images using PyMuPDF
- Generates descriptions using vision models (placeholder for now)
- Applies OCR for text-heavy figures
- Stores image paths and metadata
"""

from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field
import fitz  # PyMuPDF
from PIL import Image
import io
from loguru import logger

from .pdf_parser import BoundingBox
from config.settings import settings


@dataclass
class ExtractedFigure:
    """Represents an extracted figure with metadata."""
    figure_id: str
    doc_id: str
    page_num: int
    description: str
    image_path: str
    bbox: Optional[BoundingBox] = None
    figure_type: str = "unknown"  # chart, graph, diagram, image, etc.
    ocr_text: str = ""
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "figure_id": self.figure_id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "description": self.description,
            "image_path": self.image_path,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "figure_type": self.figure_type,
            "ocr_text": self.ocr_text,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class FigureProcessor:
    """
    Extract and process figures from PDFs.

    Features:
    - Image extraction using PyMuPDF
    - Vision model descriptions (placeholder)
    - OCR for text extraction (optional)
    - Bounding box extraction for citations
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        apply_ocr: bool = None,
        use_vision_model: bool = False  # Placeholder for Phase 5
    ):
        """
        Initialize figure processor.

        Args:
            output_dir: Directory to save extracted images
            apply_ocr: Whether to apply OCR (default from settings)
            use_vision_model: Whether to use vision model for descriptions
        """
        self.output_dir = output_dir or (settings.PROCESSED_DATA_DIR / "figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.apply_ocr = (
            apply_ocr
            if apply_ocr is not None
            else settings.APPLY_OCR_TO_FIGURES
        )

        self.use_vision_model = use_vision_model

        # Check OCR availability
        self.ocr_available = False
        if self.apply_ocr:
            try:
                import easyocr
                self.reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=settings.ENABLE_GPU)
                self.ocr_available = True
                logger.info(f"OCR enabled with languages: {settings.OCR_LANGUAGES}")
            except ImportError:
                logger.warning("EasyOCR not available - OCR will be skipped")
                self.apply_ocr = False

        logger.info(
            f"FigureProcessor initialized: output={self.output_dir}, "
            f"ocr={self.apply_ocr}, vision={self.use_vision_model}"
        )

    def extract_figures_from_pdf(
        self,
        pdf_path: Path,
        doc_id: Optional[str] = None,
        min_width: int = 100,
        min_height: int = 100
    ) -> List[ExtractedFigure]:
        """
        Extract all figures from a PDF.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier
            min_width: Minimum image width to extract
            min_height: Minimum image height to extract

        Returns:
            List of ExtractedFigure objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = doc_id or pdf_path.stem
        logger.info(f"Extracting figures from {pdf_path}")

        extracted_figures = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Get images from page
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    try:
                        figure = self._process_image(
                            doc,
                            page,
                            img,
                            page_num,
                            img_index,
                            doc_id,
                            min_width,
                            min_height
                        )

                        if figure:
                            extracted_figures.append(figure)

                    except Exception as e:
                        logger.error(f"Error processing image {img_index} on page {page_num}: {e}")
                        continue

            doc.close()

            logger.info(f"Successfully extracted {len(extracted_figures)} figures")

        except Exception as e:
            logger.error(f"Error extracting figures from {pdf_path}: {e}")

        return extracted_figures

    def _process_image(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        img: tuple,
        page_num: int,
        img_index: int,
        doc_id: str,
        min_width: int,
        min_height: int
    ) -> Optional[ExtractedFigure]:
        """
        Process a single image from the PDF.

        Args:
            doc: PyMuPDF document
            page: PyMuPDF page
            img: Image tuple from get_images()
            page_num: Page number (0-indexed)
            img_index: Image index on page
            doc_id: Document ID
            min_width: Minimum width threshold
            min_height: Minimum height threshold

        Returns:
            ExtractedFigure or None if image too small
        """
        try:
            xref = img[0]  # Image xref

            # Get image data
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Open image with PIL to check dimensions
            pil_image = Image.open(io.BytesIO(image_bytes))
            width, height = pil_image.size

            # Skip small images (likely icons or decorative elements)
            if width < min_width or height < min_height:
                logger.debug(f"Skipping small image: {width}x{height}")
                return None

            # Create figure ID and save path
            figure_id = f"{doc_id}_fig_{page_num}_{img_index}"
            image_filename = f"{figure_id}.{image_ext}"
            image_path = self.output_dir / image_filename

            # Save image
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # Get bounding box
            bbox = self._get_image_bbox(page, xref, page_num)

            # Extract OCR text if enabled
            ocr_text = ""
            if self.apply_ocr and self.ocr_available:
                ocr_text = self._extract_ocr_text(pil_image)

            # Generate description
            description = self._generate_description(
                pil_image,
                ocr_text,
                width,
                height
            )

            # Determine figure type (basic heuristic)
            figure_type = self._classify_figure_type(pil_image, ocr_text)

            figure = ExtractedFigure(
                figure_id=figure_id,
                doc_id=doc_id,
                page_num=page_num,
                description=description,
                image_path=str(image_path),
                bbox=bbox,
                figure_type=figure_type,
                ocr_text=ocr_text,
                metadata={
                    "width": width,
                    "height": height,
                    "format": image_ext,
                    "file_size": len(image_bytes)
                }
            )

            logger.debug(f"Extracted figure {figure_id}: {width}x{height}, type={figure_type}")

            return figure

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def _get_image_bbox(
        self,
        page: fitz.Page,
        xref: int,
        page_num: int
    ) -> Optional[BoundingBox]:
        """
        Get bounding box for an image.

        Args:
            page: PyMuPDF page
            xref: Image xref
            page_num: Page number

        Returns:
            BoundingBox or None
        """
        try:
            # Get image bounding box
            img_bbox = page.get_image_bbox(xref)

            return BoundingBox(
                x0=img_bbox.x0,
                y0=img_bbox.y0,
                x1=img_bbox.x1,
                y1=img_bbox.y1,
                page_num=page_num
            )

        except Exception as e:
            logger.debug(f"Could not extract bbox for image: {e}")
            return None

    def _extract_ocr_text(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR.

        Args:
            image: PIL Image

        Returns:
            Extracted text
        """
        if not self.ocr_available:
            return ""

        try:
            # Convert PIL image to numpy array
            import numpy as np
            img_array = np.array(image)

            # Run OCR
            results = self.reader.readtext(img_array, detail=0)

            # Join results
            text = " ".join(results)

            logger.debug(f"OCR extracted {len(text)} characters")

            return text.strip()

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""

    def _generate_description(
        self,
        image: Image.Image,
        ocr_text: str,
        width: int,
        height: int
    ) -> str:
        """
        Generate description for figure.

        Placeholder for vision model - currently uses rule-based approach.
        Will be enhanced with Llama 3.2 Vision in Phase 5.

        Args:
            image: PIL Image
            ocr_text: OCR extracted text
            width: Image width
            height: Image height

        Returns:
            Description string
        """
        if self.use_vision_model:
            # Placeholder for vision model integration
            # TODO: Implement in Phase 5 with Ollama Llama 3.2 Vision
            return "Vision model description (not yet implemented)"

        # Rule-based description for now
        description_parts = [
            f"Figure ({width}x{height} pixels)"
        ]

        # Add OCR text if available
        if ocr_text:
            # Truncate OCR text
            ocr_preview = ocr_text[:200] + "..." if len(ocr_text) > 200 else ocr_text
            description_parts.append(f"Text content: {ocr_preview}")

        description = ". ".join(description_parts)

        # Truncate if too long
        max_length = settings.FIGURE_DESCRIPTION_MAX_LENGTH
        if len(description) > max_length:
            description = description[:max_length-3] + "..."

        return description

    def _classify_figure_type(
        self,
        image: Image.Image,
        ocr_text: str
    ) -> str:
        """
        Classify figure type using simple heuristics.

        Args:
            image: PIL Image
            ocr_text: OCR text

        Returns:
            Figure type string
        """
        # Simple heuristics - can be improved with ML
        width, height = image.size

        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 1

        # Check if text-heavy (likely diagram/flowchart)
        if len(ocr_text) > 100:
            return "diagram"

        # Wide images might be charts/graphs
        if aspect_ratio > 1.5:
            return "chart"

        # Nearly square might be plots
        if 0.8 <= aspect_ratio <= 1.2:
            return "graph"

        # Default
        return "figure"

    def extract_figures_from_page(
        self,
        pdf_path: Path,
        page_num: int,
        doc_id: Optional[str] = None
    ) -> List[ExtractedFigure]:
        """
        Extract figures from a specific page.

        Args:
            pdf_path: Path to PDF
            page_num: Page number (0-indexed)
            doc_id: Document ID

        Returns:
            List of ExtractedFigure objects from that page
        """
        doc_id = doc_id or pdf_path.stem
        extracted_figures = []

        try:
            doc = fitz.open(pdf_path)

            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} out of range (total: {len(doc)})")

            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                try:
                    figure = self._process_image(
                        doc,
                        page,
                        img,
                        page_num,
                        img_index,
                        doc_id,
                        min_width=100,
                        min_height=100
                    )

                    if figure:
                        extracted_figures.append(figure)

                except Exception as e:
                    logger.error(f"Error processing image {img_index}: {e}")
                    continue

            doc.close()

        except Exception as e:
            logger.error(f"Error extracting figures from page {page_num}: {e}")

        return extracted_figures
