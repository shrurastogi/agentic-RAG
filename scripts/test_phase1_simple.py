"""
Simple Phase 1 test - PDF parsing and chunking only (no vector store required).

This test works without Weaviate/Docker by testing just the parsing and chunking steps.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_processor import TextProcessor
from loguru import logger


def test_parsing_and_chunking(pdf_path: Path):
    """
    Test PDF parsing and text chunking (no vector store needed).

    Args:
        pdf_path: Path to a test PDF file
    """
    logger.info("=" * 60)
    logger.info("TESTING PHASE 1: PARSING & CHUNKING (No Docker Required)")
    logger.info("=" * 60)

    # Step 1: Parse PDF
    logger.info("\n[STEP 1/3] Parsing PDF...")
    parser = PDFParser()

    try:
        parsed_doc = parser.parse_pdf(pdf_path)

        logger.info(f"✓ Successfully parsed PDF!")
        logger.info(f"\nDocument Information:")
        logger.info(f"  Title: {parsed_doc.title}")
        logger.info(f"  Document ID: {parsed_doc.doc_id}")
        logger.info(f"  Total Pages: {parsed_doc.total_pages}")
        logger.info(f"  Text Blocks: {len(parsed_doc.text_blocks)}")
        logger.info(f"  Content Regions Detected: {len(parsed_doc.content_regions)}")

        # Analyze content regions
        tables = [r for r in parsed_doc.content_regions if r.content_type == "table"]
        figures = [r for r in parsed_doc.content_regions if r.content_type == "figure"]

        logger.info(f"\nContent Analysis:")
        logger.info(f"  Detected Tables: {len(tables)}")
        logger.info(f"  Detected Figures: {len(figures)}")

        # Show sample text
        if parsed_doc.text_blocks:
            logger.info(f"\nFirst Text Block Sample:")
            first_block = parsed_doc.text_blocks[0]
            logger.info(f"  Page: {first_block.page_num}")
            logger.info(f"  Font Size: {first_block.font_size:.1f}")
            logger.info(f"  Is Header: {first_block.is_header}")
            logger.info(f"  Text: {first_block.text[:200]}...")

    except Exception as e:
        logger.error(f"✗ Failed to parse PDF: {e}")
        raise

    # Step 2: Process text into chunks
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 2/3] Processing text into chunks...")

    processor = TextProcessor()

    try:
        chunks = processor.process_document(parsed_doc)

        logger.info(f"✓ Successfully created chunks!")
        logger.info(f"\nChunk Statistics:")
        logger.info(f"  Total Chunks: {len(chunks)}")

        if chunks:
            total_tokens = sum(c.token_count for c in chunks)
            avg_tokens = total_tokens / len(chunks)
            chunks_with_stats = sum(c.metadata.get('contains_statistics', False) for c in chunks)

            logger.info(f"  Average Tokens per Chunk: {avg_tokens:.1f}")
            logger.info(f"  Total Tokens: {total_tokens}")
            logger.info(f"  Chunks with Statistics: {chunks_with_stats}")

            # Analyze sections
            sections = set(c.section_title for c in chunks if c.section_title)
            logger.info(f"  Unique Sections: {len(sections)}")
            if sections:
                logger.info(f"  Sections Found: {', '.join(list(sections)[:5])}")

            # Show sample chunks
            logger.info(f"\n📄 Sample Chunks:")
            for i, chunk in enumerate(chunks[:3], 1):
                logger.info(f"\n  Chunk {i}:")
                logger.info(f"    ID: {chunk.chunk_id}")
                logger.info(f"    Page: {chunk.page_num}")
                logger.info(f"    Section: {chunk.section_title or 'N/A'}")
                logger.info(f"    Tokens: {chunk.token_count}")
                logger.info(f"    Has Stats: {chunk.metadata.get('contains_statistics', False)}")
                logger.info(f"    Preview: {chunk.text[:150]}...")

            if len(chunks) > 3:
                logger.info(f"\n  ... and {len(chunks) - 3} more chunks")

    except Exception as e:
        logger.error(f"✗ Failed to create chunks: {e}")
        raise

    # Step 3: Summary
    logger.info("\n" + "=" * 60)
    logger.info("[STEP 3/3] Test Summary")
    logger.info("=" * 60)

    logger.info(f"\n✓ Phase 1 Core Components Working:")
    logger.info(f"  ✓ PDF Parsing - Extracted {len(parsed_doc.text_blocks)} text blocks")
    logger.info(f"  ✓ Content Detection - Found {len(tables)} tables, {len(figures)} figures")
    logger.info(f"  ✓ Text Chunking - Created {len(chunks)} semantic chunks")
    logger.info(f"  ✓ Section Awareness - Identified {len(sections)} sections")
    logger.info(f"  ✓ Statistical Detection - {chunks_with_stats} chunks contain statistics")

    logger.info(f"\n📊 Document Processing Complete!")
    logger.info(f"\n⚠️  Note: Vector store and search testing requires Weaviate")
    logger.info(f"   To test the complete pipeline with vector storage:")
    logger.info(f"   1. Install Docker Desktop")
    logger.info(f"   2. Run: docker-compose up -d")
    logger.info(f"   3. Run: python scripts/test_phase1_pipeline.py {pdf_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ PHASE 1 PARSING & CHUNKING TEST COMPLETE")
    logger.info("=" * 60)


def create_sample_content():
    """Create a simple text file if no PDF is available."""
    logger.info("\n💡 No PDF file provided.")
    logger.info("\nUsage: python scripts/test_phase1_simple.py <path_to_pdf>")
    logger.info("Example: python scripts/test_phase1_simple.py data/raw/sample.pdf")
    logger.info("\nYou can:")
    logger.info("  1. Place any PDF in data/raw/ folder")
    logger.info("  2. Download a sample: https://arxiv.org/pdf/1706.03762.pdf")
    logger.info("  3. Use any PDF file you have on your computer")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    # Check for PDF argument
    if len(sys.argv) < 2:
        create_sample_content()
        sys.exit(0)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        logger.info("\nPlease provide a valid path to a PDF file.")
        sys.exit(1)

    try:
        test_parsing_and_chunking(pdf_path)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
