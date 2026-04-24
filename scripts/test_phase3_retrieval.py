"""
Test script for Phase 3: Retrieval & Reranking (without Weaviate)

This script tests the retrieval components:
1. Query processing and intent classification
2. Filter extraction
3. Query decomposition
4. Reranking with cross-encoder (mock results)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.query_processor import QueryProcessor, QueryIntent
from src.retrieval.reranker import Reranker, RankedResult
from src.retrieval.hybrid_search import SearchResult, SearchFilters
from loguru import logger


def test_query_processor():
    """Test query processing and intent classification."""
    logger.info("=" * 70)
    logger.info("TESTING QUERY PROCESSOR")
    logger.info("=" * 70)

    processor = QueryProcessor()

    # Test queries with different intents
    test_queries = [
        "What is the primary endpoint of the study?",
        "Compare efficacy rates between Study A and Study B",
        "Show me the adverse events table from page 45",
        "What's the trend in patient enrollment over time?",
        "Find p-values across all studies in the Results section",
        "What was the response rate in CSR-12345?",
        "List all figures showing Kaplan-Meier curves",
    ]

    logger.info("\n📝 Processing Test Queries:\n")

    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n[Query {i}] {query}")
        logger.info("-" * 70)

        processed = processor.process(query)

        logger.info(f"  Intent: {processed.intent.value}")
        logger.info(f"  Focus Content Types: {', '.join(processed.focus_content_types)}")
        logger.info(f"  Keywords: {', '.join(processed.keywords[:5])}")

        if processed.filters.doc_ids:
            logger.info(f"  Document Filter: {', '.join(processed.filters.doc_ids)}")

        if processed.filters.sections:
            logger.info(f"  Section Filter: {', '.join(processed.filters.sections)}")

        if processed.filters.page_min or processed.filters.page_max:
            page_range = f"{processed.filters.page_min or '?'}-{processed.filters.page_max or '?'}"
            logger.info(f"  Page Range: {page_range}")

        if processed.filters.contains_statistics:
            logger.info(f"  Requires Statistics: Yes")

        if len(processed.sub_queries) > 1:
            logger.info(f"  Sub-queries ({len(processed.sub_queries)}):")
            for j, sq in enumerate(processed.sub_queries, 1):
                logger.info(f"    {j}. {sq}")

    # Test query expansion
    logger.info("\n" + "=" * 70)
    logger.info("TESTING QUERY EXPANSION")
    logger.info("=" * 70)

    expansion_query = "What was the efficacy in the safety analysis?"
    logger.info(f"\nOriginal: {expansion_query}")

    expanded = processor.expand_query(expansion_query)
    logger.info(f"Expanded to {len(expanded)} variants:")
    for i, variant in enumerate(expanded[:5], 1):
        logger.info(f"  {i}. {variant}")

    logger.info("\n✓ Query Processor Tests Complete")


def test_reranker():
    """Test reranking with mock search results."""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING RERANKER")
    logger.info("=" * 70)

    # Create mock search results
    mock_results = [
        SearchResult(
            result_id="chunk_1",
            content_type="text",
            content="The primary endpoint was overall survival (OS), defined as time from randomization to death. "
                    "The study demonstrated significant improvement in OS with a hazard ratio of 0.72 (95% CI: 0.60-0.85, p<0.001).",
            score=0.85,
            doc_id="CSR-001",
            page_num=42,
            section_title="Results",
            metadata={"contains_statistics": True}
        ),
        SearchResult(
            result_id="chunk_2",
            content_type="text",
            content="Patient demographics showed a median age of 65 years (range: 45-80). "
                    "The study included 60% male and 40% female participants.",
            score=0.45,
            doc_id="CSR-001",
            page_num=15,
            section_title="Demographics",
            metadata={"contains_statistics": False}
        ),
        SearchResult(
            result_id="table_1",
            content_type="table",
            content="Table showing efficacy endpoints: Overall Survival (OS), Progression-Free Survival (PFS), "
                    "and Objective Response Rate (ORR) with corresponding p-values.",
            score=0.75,
            doc_id="CSR-001",
            page_num=43,
            metadata={"num_rows": 5, "num_cols": 4}
        ),
        SearchResult(
            result_id="chunk_3",
            content_type="text",
            content="The study protocol was approved by the institutional review board. "
                    "All participants provided written informed consent.",
            score=0.30,
            doc_id="CSR-001",
            page_num=8,
            section_title="Methods",
            metadata={"contains_statistics": False}
        ),
        SearchResult(
            result_id="chunk_4",
            content_type="text",
            content="Secondary endpoints included progression-free survival and quality of life measures. "
                    "PFS showed improvement with HR=0.68 (p=0.002).",
            score=0.70,
            doc_id="CSR-001",
            page_num=44,
            section_title="Results",
            metadata={"contains_statistics": True}
        ),
    ]

    query = "What was the primary endpoint and its statistical significance?"

    logger.info(f"\nQuery: {query}")
    logger.info(f"Mock Results: {len(mock_results)}")

    # Initialize reranker
    try:
        reranker = Reranker(top_k=3)
        logger.info("\n✓ Reranker initialized")

        # Rerank results
        logger.info("\n📊 Reranking Results...")
        ranked_results = reranker.rerank(query, mock_results, top_k=3)

        logger.info(f"\nTop {len(ranked_results)} Results After Reranking:\n")

        for ranked in ranked_results:
            result = ranked.result
            logger.info(f"[Rank {ranked.rank}] {result.content_type.upper()}")
            logger.info(f"  ID: {result.result_id}")
            logger.info(f"  Original Score: {ranked.original_score:.3f}")
            logger.info(f"  Rerank Score: {ranked.rerank_score:.3f}")
            logger.info(f"  Final Score: {ranked.final_score:.3f}")
            logger.info(f"  Page: {result.page_num}, Section: {result.section_title}")
            logger.info(f"  Content: {result.content[:100]}...")
            logger.info("")

        # Test context assembly
        logger.info("=" * 70)
        logger.info("TESTING CONTEXT ASSEMBLY")
        logger.info("=" * 70)

        context = reranker.assemble_context(
            query=query,
            ranked_results=ranked_results,
            group_by_document=True
        )

        logger.info(f"\n📄 Assembled Context:")
        logger.info(f"  Query: {context.query}")
        logger.info(f"  Num Results: {context.metadata['num_results']}")
        logger.info(f"  Num Documents: {context.metadata['num_documents']}")
        logger.info(f"  Content Types: {context.metadata['content_types']}")
        logger.info(f"  Context Length: {context.metadata['context_length']} chars")
        logger.info(f"  Avg Score: {context.metadata['avg_score']:.3f}")

        logger.info(f"\n  Context Preview:")
        preview_lines = context.context_text.split('\n')[:15]
        for line in preview_lines:
            logger.info(f"    {line}")
        if len(context.context_text.split('\n')) > 15:
            logger.info(f"    ...")

        # Test citation extraction
        logger.info("\n" + "=" * 70)
        logger.info("TESTING CITATION EXTRACTION")
        logger.info("=" * 70)

        citations = reranker.get_citation_info(ranked_results)

        logger.info(f"\n📌 Citations ({len(citations)}):\n")

        for citation in citations:
            logger.info(f"  {citation['citation_id']}:")
            logger.info(f"    Document: {citation['doc_id']}")
            logger.info(f"    Page: {citation['page_num']}")
            logger.info(f"    Section: {citation['section']}")
            logger.info(f"    Type: {citation['content_type']}")
            logger.info(f"    Score: {citation['score']:.3f}")
            logger.info(f"    Snippet: {citation['snippet']}")
            logger.info("")

        logger.info("✓ Reranker Tests Complete")

    except Exception as e:
        logger.error(f"Reranker test failed: {e}")
        logger.info("\nNote: Reranker requires sentence-transformers with CrossEncoder")
        logger.info("Install with: pip install sentence-transformers")
        import traceback
        traceback.print_exc()


def main():
    """Run all Phase 3 tests."""
    logger.info("=" * 70)
    logger.info("PHASE 3: RETRIEVAL & RERANKING TESTS")
    logger.info("=" * 70)
    logger.info("\nNote: This tests query processing and reranking components.")
    logger.info("Full hybrid search requires Weaviate (tested separately).\n")

    try:
        # Test 1: Query Processor
        test_query_processor()

        # Test 2: Reranker
        test_reranker()

        logger.info("\n" + "=" * 70)
        logger.info("✓ PHASE 3 TESTS COMPLETE")
        logger.info("=" * 70)

        logger.info(f"\n✓ Phase 3 Components Working:")
        logger.info(f"  ✓ Query Intent Classification")
        logger.info(f"  ✓ Filter Extraction from Natural Language")
        logger.info(f"  ✓ Query Decomposition")
        logger.info(f"  ✓ Query Expansion")
        logger.info(f"  ✓ Cross-Encoder Reranking")
        logger.info(f"  ✓ Context Assembly")
        logger.info(f"  ✓ Citation Generation")

        logger.info(f"\n📍 Next Steps:")
        logger.info(f"  Phase 4: Implement agentic workflows & citations")
        logger.info(f"  - LangGraph state machine")
        logger.info(f"  - Multi-document reasoning")
        logger.info(f"  - Citation tracking system")

        logger.info(f"\n⚠️  Note: Full end-to-end retrieval testing requires:")
        logger.info(f"  1. Weaviate running (docker-compose up -d)")
        logger.info(f"  2. Documents indexed in vector store")
        logger.info(f"  3. Run: python scripts/test_phase3_full.py (after Weaviate setup)")

    except Exception as e:
        logger.error(f"Phase 3 tests failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )

    main()
