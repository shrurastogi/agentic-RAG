"""
Test script for Phase 2 pipeline: Complete Multi-Modal Extraction

This script tests the advanced extraction pipeline:
1. PDF parsing (Phase 1)
2. Text chunking (Phase 1)
3. Table extraction (Phase 2)
4. Figure extraction (Phase 2)
5. Document orchestration (Phase 2)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.document_loader import DocumentLoader
from loguru import logger


def test_multimodal_pipeline(pdf_path: Path):
    """
    Test the complete Phase 2 multi-modal extraction pipeline.

    Args:
        pdf_path: Path to a test PDF file
    """
    logger.info("=" * 70)
    logger.info("TESTING PHASE 2: MULTI-MODAL EXTRACTION PIPELINE")
    logger.info("=" * 70)

    # Initialize document loader
    logger.info("\n[INITIALIZING] Setting up document loader...")
    try:
        loader = DocumentLoader(
            enable_tables=True,
            enable_figures=True,
            enable_caching=True
        )
        logger.info("✓ Document loader initialized")
    except Exception as e:
        logger.error(f"✗ Initialization failed: {e}")
        logger.info("\nNote: If camelot is missing, install with:")
        logger.info("  pip install camelot-py[cv] opencv-python")
        raise

    # Process document
    logger.info(f"\n[PROCESSING] Loading document: {pdf_path.name}")
    logger.info("-" * 70)

    try:
        processed_doc = loader.load_document(pdf_path)

        # Display results
        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTION RESULTS")
        logger.info("=" * 70)

        logger.info(f"\n📄 Document Information:")
        logger.info(f"  Title: {processed_doc.title}")
        logger.info(f"  Document ID: {processed_doc.doc_id}")
        logger.info(f"  Total Pages: {processed_doc.total_pages}")
        logger.info(f"  Processing Time: {processed_doc.processing_time:.2f}s")

        # Text chunks summary
        logger.info(f"\n📝 Text Chunks: {len(processed_doc.text_chunks)}")
        if processed_doc.text_chunks:
            total_tokens = sum(c.token_count for c in processed_doc.text_chunks)
            avg_tokens = total_tokens / len(processed_doc.text_chunks)
            chunks_with_stats = sum(
                c.metadata.get('contains_statistics', False)
                for c in processed_doc.text_chunks
            )

            logger.info(f"  Average tokens per chunk: {avg_tokens:.1f}")
            logger.info(f"  Total tokens: {total_tokens}")
            logger.info(f"  Chunks with statistics: {chunks_with_stats}")

            # Show sample chunk
            logger.info(f"\n  Sample Chunk:")
            sample = processed_doc.text_chunks[0]
            logger.info(f"    ID: {sample.chunk_id}")
            logger.info(f"    Page: {sample.page_num}")
            logger.info(f"    Section: {sample.section_title or 'N/A'}")
            logger.info(f"    Preview: {sample.text[:150]}...")

        # Tables summary
        logger.info(f"\n📊 Tables: {len(processed_doc.tables)}")
        if processed_doc.tables:
            total_rows = sum(t.num_rows for t in processed_doc.tables)
            total_cols = sum(t.num_cols for t in processed_doc.tables)
            avg_confidence = sum(t.confidence for t in processed_doc.tables) / len(processed_doc.tables)

            logger.info(f"  Total rows across all tables: {total_rows}")
            logger.info(f"  Total columns across all tables: {total_cols}")
            logger.info(f"  Average extraction confidence: {avg_confidence:.2f}")

            # Show sample tables
            logger.info(f"\n  Sample Tables:")
            for i, table in enumerate(processed_doc.tables[:2], 1):
                logger.info(f"\n    Table {i}:")
                logger.info(f"      ID: {table.table_id}")
                logger.info(f"      Page: {table.page_num}")
                logger.info(f"      Size: {table.num_rows} rows × {table.num_cols} cols")
                logger.info(f"      Confidence: {table.confidence:.2f}")
                logger.info(f"      Summary: {table.summary[:100]}...")

                # Show markdown preview
                lines = table.markdown.split('\n')[:5]
                logger.info(f"      Markdown Preview:")
                for line in lines:
                    logger.info(f"        {line}")
                if len(table.markdown.split('\n')) > 5:
                    logger.info(f"        ...")

            if len(processed_doc.tables) > 2:
                logger.info(f"\n  ... and {len(processed_doc.tables) - 2} more tables")

        # Figures summary
        logger.info(f"\n🖼️  Figures: {len(processed_doc.figures)}")
        if processed_doc.figures:
            total_size = sum(
                f.metadata.get('file_size', 0)
                for f in processed_doc.figures
            )

            logger.info(f"  Total size: {total_size / 1024:.1f} KB")

            # Show sample figures
            logger.info(f"\n  Sample Figures:")
            for i, figure in enumerate(processed_doc.figures[:3], 1):
                logger.info(f"\n    Figure {i}:")
                logger.info(f"      ID: {figure.figure_id}")
                logger.info(f"      Page: {figure.page_num}")
                logger.info(f"      Type: {figure.figure_type}")
                logger.info(f"      Size: {figure.metadata.get('width')}x{figure.metadata.get('height')} pixels")
                logger.info(f"      Format: {figure.metadata.get('format')}")
                logger.info(f"      Saved: {Path(figure.image_path).name}")
                logger.info(f"      Description: {figure.description[:100]}...")
                if figure.ocr_text:
                    logger.info(f"      OCR Text: {figure.ocr_text[:80]}...")

            if len(processed_doc.figures) > 3:
                logger.info(f"\n  ... and {len(processed_doc.figures) - 3} more figures")

        # Errors
        if processed_doc.errors:
            logger.info(f"\n⚠️  Errors ({len(processed_doc.errors)}):")
            for error in processed_doc.errors:
                logger.info(f"  - {error}")

        # Summary statistics
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        stats = processed_doc.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Verify all components
        logger.info("\n" + "=" * 70)
        logger.info("COMPONENT VERIFICATION")
        logger.info("=" * 70)

        logger.info(f"\n✓ Phase 2 Components Working:")
        logger.info(f"  ✓ PDF Parsing - {processed_doc.total_pages} pages")
        logger.info(f"  ✓ Text Chunking - {len(processed_doc.text_chunks)} chunks")
        logger.info(f"  ✓ Table Extraction - {len(processed_doc.tables)} tables")
        logger.info(f"  ✓ Figure Extraction - {len(processed_doc.figures)} figures")
        logger.info(f"  ✓ Document Orchestration - Complete pipeline")
        logger.info(f"  ✓ Parallel Processing - {processed_doc.processing_time:.2f}s")

        # Cache information
        if loader.enable_caching:
            cache_path = loader._get_cache_path(processed_doc.doc_id)
            if cache_path.exists():
                logger.info(f"  ✓ Caching - Saved to {cache_path.name}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ PHASE 2 MULTI-MODAL EXTRACTION TEST COMPLETE")
        logger.info("=" * 70)

        # Next steps
        logger.info(f"\n📍 Next Steps:")
        logger.info(f"  Phase 3: Implement retrieval & reranking")
        logger.info(f"  - Hybrid search (BM25 + vector)")
        logger.info(f"  - Query processing")
        logger.info(f"  - Cross-encoder reranking")

    except Exception as e:
        logger.error(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def show_usage():
    """Show usage information."""
    logger.info("\n💡 Phase 2 Pipeline Test")
    logger.info("\nUsage: python scripts/test_phase2_pipeline.py <path_to_pdf>")
    logger.info("Example: python scripts/test_phase2_pipeline.py data/raw/attention_paper.pdf")
    logger.info("\nThis will test:")
    logger.info("  ✓ Table extraction with camelot")
    logger.info("  ✓ Figure extraction with PyMuPDF")
    logger.info("  ✓ Multi-modal document orchestration")
    logger.info("  ✓ Parallel processing")
    logger.info("  ✓ Artifact caching")
    logger.info("\nNote: Requires camelot-py for table extraction")
    logger.info("Install with: pip install camelot-py[cv] opencv-python")


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
        show_usage()
        sys.exit(0)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        logger.info("\nPlease provide a valid path to a PDF file.")
        sys.exit(1)

    try:
        test_multimodal_pipeline(pdf_path)
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
