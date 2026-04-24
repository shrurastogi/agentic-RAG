"""
Agent Tools for RAG system.

Provides tools that agents can use:
- search_documents: Search with filters
- retrieve_table: Get specific table
- retrieve_figure: Get figure with description
- compare_across_docs: Multi-document comparison
- extract_statistics: Pull numerical data
- verify_citation: Verify claims against sources
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import re
from loguru import logger

from src.retrieval.hybrid_search import HybridSearch, SearchFilters, SearchResult
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.reranker import Reranker


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    message: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AgentTools:
    """
    Collection of tools for agent workflows.

    Tools provide specific capabilities that agents can use
    to accomplish tasks.
    """

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize agent tools.

        Args:
            hybrid_search: HybridSearch instance
            query_processor: QueryProcessor instance
            reranker: Reranker instance
        """
        self.hybrid_search = hybrid_search or HybridSearch()
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker or Reranker()

        logger.info("AgentTools initialized")

    def search_documents(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
        page_range: Optional[tuple] = None,
        content_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> ToolResult:
        """
        Search documents with optional filters.

        Args:
            query: Search query
            doc_ids: Filter by document IDs
            sections: Filter by sections
            page_range: Tuple of (min_page, max_page)
            content_types: Filter by content type ["text", "table", "figure"]
            limit: Maximum results

        Returns:
            ToolResult with search results
        """
        try:
            logger.info(f"search_documents: '{query[:50]}...'")

            # Build filters
            filters = SearchFilters(
                doc_ids=doc_ids,
                sections=sections,
                page_min=page_range[0] if page_range else None,
                page_max=page_range[1] if page_range else None,
                content_types=content_types
            )

            # Perform search
            results = self.hybrid_search.search(
                query=query,
                filters=filters,
                include_text="text" in (content_types or ["text", "table", "figure"]),
                include_tables="table" in (content_types or ["text", "table", "figure"]),
                include_figures="figure" in (content_types or ["text", "table", "figure"])
            )

            # Limit results
            results = results[:limit]

            return ToolResult(
                success=True,
                data=results,
                message=f"Found {len(results)} results",
                metadata={
                    "num_results": len(results),
                    "query": query,
                    "filters": filters.__dict__
                }
            )

        except Exception as e:
            logger.error(f"search_documents failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                message=f"Search failed: {e}"
            )

    def retrieve_table(
        self,
        query: str,
        doc_id: Optional[str] = None,
        page_num: Optional[int] = None,
        limit: int = 3
    ) -> ToolResult:
        """
        Retrieve specific tables.

        Args:
            query: Description of table needed
            doc_id: Optional document filter
            page_num: Optional page filter
            limit: Maximum tables to return

        Returns:
            ToolResult with table data
        """
        try:
            logger.info(f"retrieve_table: '{query[:50]}...'")

            # Search tables only
            filters = SearchFilters(
                doc_ids=[doc_id] if doc_id else None,
                page_min=page_num,
                page_max=page_num
            )

            results = self.hybrid_search.search_tables_only(
                query=query,
                filters=filters,
                limit=limit
            )

            # Extract table-specific data
            tables = []
            for result in results:
                table_data = {
                    "table_id": result.result_id,
                    "doc_id": result.doc_id,
                    "page_num": result.page_num,
                    "summary": result.content,
                    "markdown": result.metadata.get("markdown", ""),
                    "json_structure": result.metadata.get("json_structure", ""),
                    "num_rows": result.metadata.get("num_rows", 0),
                    "num_cols": result.metadata.get("num_cols", 0),
                    "score": result.score
                }
                tables.append(table_data)

            return ToolResult(
                success=True,
                data=tables,
                message=f"Retrieved {len(tables)} tables",
                metadata={"num_tables": len(tables)}
            )

        except Exception as e:
            logger.error(f"retrieve_table failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                message=f"Table retrieval failed: {e}"
            )

    def retrieve_figure(
        self,
        query: str,
        doc_id: Optional[str] = None,
        figure_type: Optional[str] = None,
        limit: int = 3
    ) -> ToolResult:
        """
        Retrieve figures/images.

        Args:
            query: Description of figure needed
            doc_id: Optional document filter
            figure_type: Optional figure type filter
            limit: Maximum figures

        Returns:
            ToolResult with figure data
        """
        try:
            logger.info(f"retrieve_figure: '{query[:50]}...'")

            filters = SearchFilters(
                doc_ids=[doc_id] if doc_id else None
            )

            results = self.hybrid_search.search_figures_only(
                query=query,
                filters=filters,
                limit=limit
            )

            # Extract figure-specific data
            figures = []
            for result in results:
                if figure_type and result.metadata.get("figure_type") != figure_type:
                    continue

                figure_data = {
                    "figure_id": result.result_id,
                    "doc_id": result.doc_id,
                    "page_num": result.page_num,
                    "description": result.content,
                    "image_path": result.metadata.get("image_path", ""),
                    "figure_type": result.metadata.get("figure_type", ""),
                    "ocr_text": result.metadata.get("ocr_text", ""),
                    "score": result.score
                }
                figures.append(figure_data)

            return ToolResult(
                success=True,
                data=figures,
                message=f"Retrieved {len(figures)} figures",
                metadata={"num_figures": len(figures)}
            )

        except Exception as e:
            logger.error(f"retrieve_figure failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                message=f"Figure retrieval failed: {e}"
            )

    def compare_across_docs(
        self,
        query: str,
        doc_ids: List[str],
        aspect: str = "general"
    ) -> ToolResult:
        """
        Compare information across multiple documents.

        Args:
            query: What to compare
            doc_ids: List of document IDs to compare
            aspect: Aspect to compare (general, efficacy, safety, etc.)

        Returns:
            ToolResult with comparison data
        """
        try:
            logger.info(f"compare_across_docs: {len(doc_ids)} docs, aspect='{aspect}'")

            # Search each document separately
            doc_results = {}

            for doc_id in doc_ids:
                filters = SearchFilters(doc_ids=[doc_id])

                results = self.hybrid_search.search(
                    query=query,
                    filters=filters,
                    include_text=True,
                    include_tables=True,
                    include_figures=False  # Focus on text/tables for comparison
                )

                doc_results[doc_id] = results[:5]  # Top 5 per doc

            # Organize by document
            comparison_data = {
                "query": query,
                "aspect": aspect,
                "documents": {}
            }

            for doc_id, results in doc_results.items():
                comparison_data["documents"][doc_id] = {
                    "num_results": len(results),
                    "top_results": [
                        {
                            "content": r.content,
                            "page_num": r.page_num,
                            "section": r.section_title,
                            "content_type": r.content_type,
                            "score": r.score
                        }
                        for r in results
                    ]
                }

            return ToolResult(
                success=True,
                data=comparison_data,
                message=f"Compared {len(doc_ids)} documents",
                metadata={"num_documents": len(doc_ids)}
            )

        except Exception as e:
            logger.error(f"compare_across_docs failed: {e}")
            return ToolResult(
                success=False,
                data={},
                message=f"Comparison failed: {e}"
            )

    def extract_statistics(
        self,
        query: str,
        doc_id: Optional[str] = None,
        stat_type: Optional[str] = None
    ) -> ToolResult:
        """
        Extract statistical data.

        Args:
            query: What statistics to find
            doc_id: Optional document filter
            stat_type: Type of statistic (p-value, CI, mean, etc.)

        Returns:
            ToolResult with statistical data
        """
        try:
            logger.info(f"extract_statistics: '{query[:50]}...'")

            # Search with statistical content filter
            filters = SearchFilters(
                doc_ids=[doc_id] if doc_id else None,
                contains_statistics=True
            )

            results = self.hybrid_search.search(
                query=query,
                filters=filters,
                include_text=True,
                include_tables=True,
                include_figures=False
            )

            # Extract statistics from content
            statistics = []

            for result in results[:10]:  # Top 10 results
                # Extract numbers and statistical patterns
                stats = self._extract_stat_patterns(result.content, stat_type)

                if stats:
                    statistics.append({
                        "source": result.result_id,
                        "doc_id": result.doc_id,
                        "page_num": result.page_num,
                        "section": result.section_title,
                        "statistics": stats,
                        "context": result.content[:200]
                    })

            return ToolResult(
                success=True,
                data=statistics,
                message=f"Extracted statistics from {len(statistics)} sources",
                metadata={"num_sources": len(statistics)}
            )

        except Exception as e:
            logger.error(f"extract_statistics failed: {e}")
            return ToolResult(
                success=False,
                data=[],
                message=f"Statistics extraction failed: {e}"
            )

    def verify_citation(
        self,
        claim: str,
        source_id: str,
        source_content: str
    ) -> ToolResult:
        """
        Verify a claim against source content.

        Args:
            claim: The claim to verify
            source_id: Source identifier
            source_content: Content to verify against

        Returns:
            ToolResult with verification result
        """
        try:
            logger.info(f"verify_citation: claim='{claim[:50]}...'")

            # Simple keyword-based verification
            # In production, would use LLM for semantic verification
            claim_lower = claim.lower()
            content_lower = source_content.lower()

            # Extract key terms from claim
            claim_words = set(re.findall(r'\b\w+\b', claim_lower))
            content_words = set(re.findall(r'\b\w+\b', content_lower))

            # Calculate overlap
            overlap = claim_words & content_words
            coverage = len(overlap) / len(claim_words) if claim_words else 0

            verified = coverage > 0.5  # >50% word overlap

            verification_result = {
                "verified": verified,
                "confidence": coverage,
                "source_id": source_id,
                "claim": claim,
                "matched_terms": list(overlap)[:10],
                "explanation": (
                    f"Claim verified with {coverage:.1%} term overlap"
                    if verified
                    else f"Insufficient overlap ({coverage:.1%}) to verify claim"
                )
            }

            return ToolResult(
                success=True,
                data=verification_result,
                message="Verification complete",
                metadata={"verified": verified, "confidence": coverage}
            )

        except Exception as e:
            logger.error(f"verify_citation failed: {e}")
            return ToolResult(
                success=False,
                data={"verified": False},
                message=f"Verification failed: {e}"
            )

    def _extract_stat_patterns(
        self,
        text: str,
        stat_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract statistical patterns from text.

        Args:
            text: Text to extract from
            stat_type: Type of statistic to focus on

        Returns:
            List of extracted statistics
        """
        statistics = []

        # P-value patterns
        if not stat_type or stat_type == "p-value":
            p_values = re.findall(
                r'p\s*[<>=]\s*0?\.\d+',
                text,
                re.IGNORECASE
            )
            for pval in p_values:
                statistics.append({"type": "p-value", "value": pval})

        # Confidence interval patterns
        if not stat_type or stat_type == "CI":
            ci_patterns = re.findall(
                r'\d+%?\s*(?:CI|confidence interval)\s*[\[\(]([^)\]]+)[\]\)]',
                text,
                re.IGNORECASE
            )
            for ci in ci_patterns:
                statistics.append({"type": "confidence_interval", "value": ci})

        # Hazard ratio / Odds ratio patterns
        if not stat_type or stat_type in ["HR", "OR"]:
            ratio_patterns = re.findall(
                r'(?:hazard ratio|HR|odds ratio|OR)\s*[=:]\s*([\d\.]+)',
                text,
                re.IGNORECASE
            )
            for ratio in ratio_patterns:
                statistics.append({"type": "ratio", "value": ratio})

        # Percentage patterns
        if not stat_type or stat_type == "percentage":
            percentages = re.findall(
                r'\b(\d+\.?\d*)\s*%',
                text
            )
            for pct in percentages[:5]:  # Limit to avoid noise
                statistics.append({"type": "percentage", "value": f"{pct}%"})

        return statistics
