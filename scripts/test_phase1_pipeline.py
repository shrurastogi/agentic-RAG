"""
Test script for Phase 1 pipeline: PDF → chunks → vector store

This script tests the basic ingestion pipeline:
1. Parse a PDF file
2. Chunk the text
3. Insert into Weaviate
4. Perform a basic search
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.text_processor import TextProcessor
from src.embeddings.vector_store import VectorStore
from loguru import logger
from config.settings import settings


def test_pipeline(pdf_path: Path):
    """
    Test the complete Phase 1 pipeline.

    Args:
        pdf_path: Path to a test PDF file
    """
    logger.info("=" * 60)
    logger.info("TESTING PHASE 1 PIPELINE")
    logger.info("=" * 60)

    # Step 1: Parse PDF
    logger.info("\n[STEP 1] Parsing PDF...")
    parser = PDFParser()
    parsed_doc = parser.parse_pdf(pdf_path)

    logger.info(f"✓ Parsed: {parsed_doc.title}")
    logger.info(f"  - Total pages: {parsed_doc.total_pages}")
    logger.info(f"  - Text blocks: {len(parsed_doc.text_blocks)}")
    logger.info(f"  - Content regions: {len(parsed_doc.content_regions)}")

    # Step 2: Process text into chunks
    logger.info("\n[STEP 2] Processing text into chunks...")
    processor = TextProcessor()
    chunks = processor.process_document(parsed_doc)

    logger.info(f"✓ Created {len(chunks)} chunks")
    if chunks:
        logger.info(f"  - First chunk: {chunks[0].chunk_id}")
        logger.info(f"  - Average tokens per chunk: {sum(c.token_count for c in chunks) / len(chunks):.1f}")
        logger.info(f"  - Chunks with statistics: {sum(c.metadata.get('contains_statistics', False) for c in chunks)}")

    # Step 3: Connect to Weaviate and create schema
    logger.info("\n[STEP 3] Connecting to Weaviate...")
    vector_store = VectorStore()

    logger.info("✓ Connected to Weaviate")

    logger.info("\n[STEP 4] Creating schema...")
    vector_store.create_schema()
    logger.info("✓ Schema created")

    # Step 4: Insert chunks
    logger.info("\n[STEP 5] Inserting chunks into vector store...")
    vector_store.insert_text_chunks(chunks)
    logger.info("✓ Chunks inserted")

    # Step 5: Verify insertion
    logger.info("\n[STEP 6] Verifying insertion...")
    stats = vector_store.get_collection_stats()
    logger.info(f"✓ Collection stats:")
    for class_name, count in stats.items():
        logger.info(f"  - {class_name}: {count} entries")

    # Step 6: Test search
    logger.info("\n[STEP 7] Testing search...")
    test_queries = [
        "What is the main topic of this document?",
        "statistical analysis",
        "methodology"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = vector_store.search_text_chunks(query, limit=3)

        if results:
            for i, result in enumerate(results, 1):
                logger.info(f"  Result {i}:")
                logger.info(f"    - Chunk: {result['chunk_id']}")
                logger.info(f"    - Score: {result['score']:.4f}")
                logger.info(f"    - Page: {result['page_num']}")
                logger.info(f"    - Section: {result.get('section_title', 'N/A')}")
                logger.info(f"    - Preview: {result['text'][:100]}...")
        else:
            logger.warning("  No results found")

    # Cleanup
    vector_store.close()

    logger.info("\n" + "=" * 60)
    logger.info("✓ PHASE 1 PIPELINE TEST COMPLETE")
    logger.info("=" * 60)


def create_sample_pdf():
    """Create a simple sample PDF for testing if none exists."""
    logger.warning("No test PDF provided. Please provide a PDF file path.")
    logger.info("\nUsage: python scripts/test_phase1_pipeline.py <path_to_pdf>")
    logger.info("Example: python scripts/test_phase1_pipeline.py data/raw/sample.pdf")


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
        create_sample_pdf()
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        test_pipeline(pdf_path)
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
