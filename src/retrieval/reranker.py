"""
Reranker using cross-encoder models for improved relevance scoring.

This module provides:
- Cross-encoder reranking
- Context assembly
- Citation preparation
- Result grouping by document
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    logger.warning("sentence-transformers CrossEncoder not available")

from .hybrid_search import SearchResult
from config.settings import settings


@dataclass
class RankedResult:
    """Represents a reranked result with improved score."""
    result: SearchResult
    original_score: float
    rerank_score: float
    final_score: float
    rank: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            **self.result.to_dict(),
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "rank": self.rank
        }


@dataclass
class RetrievalContext:
    """Assembled context from retrieval results."""
    query: str
    ranked_results: List[RankedResult]
    context_text: str
    documents: Dict[str, List[RankedResult]]  # Grouped by doc_id
    metadata: Dict = field(default_factory=dict)


class Reranker:
    """
    Rerank search results using cross-encoder for better relevance.

    Features:
    - Cross-encoder reranking
    - Score normalization
    - Context assembly
    - Document grouping
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        enable_reranking: Optional[bool] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name (default from settings)
            enable_reranking: Whether to enable reranking (default from settings)
            top_k: Number of results to keep after reranking
        """
        self.model_name = model_name or settings.RERANKER_MODEL
        self.enable_reranking = (
            enable_reranking
            if enable_reranking is not None
            else settings.ENABLE_RERANKING
        )
        self.top_k = top_k or settings.TOP_K_RERANK

        self.model = None

        if self.enable_reranking:
            if not CROSSENCODER_AVAILABLE:
                logger.warning(
                    "Cross-encoder not available - reranking disabled. "
                    "Install with: pip install sentence-transformers"
                )
                self.enable_reranking = False
            else:
                try:
                    logger.info(f"Loading reranker model: {self.model_name}")
                    self.model = CrossEncoder(self.model_name)
                    logger.info("Reranker loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load reranker: {e}")
                    self.enable_reranking = False

        logger.info(
            f"Reranker initialized: enabled={self.enable_reranking}, "
            f"top_k={self.top_k}"
        )

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[RankedResult]:
        """
        Rerank search results.

        Args:
            query: Original query
            results: Search results from retrieval
            top_k: Number of results to return (default: self.top_k)

        Returns:
            List of RankedResult objects, sorted by final score
        """
        if not results:
            return []

        top_k = top_k or self.top_k

        logger.info(f"Reranking {len(results)} results (top_k={top_k})")

        if not self.enable_reranking or not self.model:
            # No reranking - just use original scores
            return self._create_ranked_results_no_rerank(results, top_k)

        # Prepare query-document pairs
        pairs = [(query, result.content) for result in results]

        try:
            # Get cross-encoder scores
            rerank_scores = self.model.predict(pairs)

            # Convert to list if numpy array
            if isinstance(rerank_scores, np.ndarray):
                rerank_scores = rerank_scores.tolist()

            # Create ranked results
            ranked_results = []
            for i, (result, rerank_score) in enumerate(zip(results, rerank_scores)):
                # Combine original and rerank scores
                # Higher weight on rerank score (0.7) vs original (0.3)
                final_score = 0.3 * result.score + 0.7 * float(rerank_score)

                ranked = RankedResult(
                    result=result,
                    original_score=result.score,
                    rerank_score=float(rerank_score),
                    final_score=final_score,
                    rank=0  # Will be set after sorting
                )
                ranked_results.append(ranked)

            # Sort by final score (descending)
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)

            # Assign ranks and take top k
            for i, ranked in enumerate(ranked_results[:top_k], 1):
                ranked.rank = i

            logger.info(f"Reranked to top {len(ranked_results[:top_k])} results")

            return ranked_results[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to no reranking
            return self._create_ranked_results_no_rerank(results, top_k)

    def _create_ranked_results_no_rerank(
        self,
        results: List[SearchResult],
        top_k: int
    ) -> List[RankedResult]:
        """
        Create ranked results without reranking (using original scores).

        Args:
            results: Search results
            top_k: Number to keep

        Returns:
            List of RankedResult objects
        """
        # Sort by original score
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        ranked_results = []
        for i, result in enumerate(sorted_results[:top_k], 1):
            ranked = RankedResult(
                result=result,
                original_score=result.score,
                rerank_score=result.score,  # Same as original
                final_score=result.score,
                rank=i
            )
            ranked_results.append(ranked)

        return ranked_results

    def assemble_context(
        self,
        query: str,
        ranked_results: List[RankedResult],
        max_length: Optional[int] = None,
        group_by_document: bool = True
    ) -> RetrievalContext:
        """
        Assemble context from ranked results.

        Args:
            query: Original query
            ranked_results: Reranked results
            max_length: Maximum context length in characters
            group_by_document: Whether to group results by document

        Returns:
            RetrievalContext object
        """
        logger.debug(f"Assembling context from {len(ranked_results)} results")

        # Group by document
        documents = {}
        for ranked in ranked_results:
            doc_id = ranked.result.doc_id
            if doc_id not in documents:
                documents[doc_id] = []
            documents[doc_id].append(ranked)

        # Build context text
        context_parts = []

        if group_by_document:
            # Group by document
            for doc_id, doc_results in documents.items():
                context_parts.append(f"[Document: {doc_id}]")

                for ranked in doc_results:
                    result = ranked.result
                    context_parts.append(
                        f"[{result.content_type.upper()} | Page {result.page_num} | "
                        f"Score: {ranked.final_score:.3f}]"
                    )

                    if result.section_title:
                        context_parts.append(f"Section: {result.section_title}")

                    context_parts.append(result.content)
                    context_parts.append("")  # Blank line

        else:
            # Flat list
            for ranked in ranked_results:
                result = ranked.result
                context_parts.append(
                    f"[{result.content_type.upper()} | Doc: {result.doc_id} | "
                    f"Page {result.page_num} | Score: {ranked.final_score:.3f}]"
                )

                if result.section_title:
                    context_parts.append(f"Section: {result.section_title}")

                context_parts.append(result.content)
                context_parts.append("")  # Blank line

        context_text = "\n".join(context_parts)

        # Truncate if needed
        if max_length and len(context_text) > max_length:
            context_text = context_text[:max_length] + "\n\n[...truncated]"

        # Compute statistics
        content_type_counts = {}
        for ranked in ranked_results:
            ctype = ranked.result.content_type
            content_type_counts[ctype] = content_type_counts.get(ctype, 0) + 1

        retrieval_context = RetrievalContext(
            query=query,
            ranked_results=ranked_results,
            context_text=context_text,
            documents=documents,
            metadata={
                "num_results": len(ranked_results),
                "num_documents": len(documents),
                "content_types": content_type_counts,
                "context_length": len(context_text),
                "avg_score": np.mean([r.final_score for r in ranked_results]) if ranked_results else 0.0
            }
        )

        logger.info(
            f"Context assembled: {len(ranked_results)} results, "
            f"{len(documents)} documents, {len(context_text)} chars"
        )

        return retrieval_context

    def get_citation_info(
        self,
        ranked_results: List[RankedResult]
    ) -> List[Dict]:
        """
        Extract citation information from results.

        Args:
            ranked_results: Reranked results

        Returns:
            List of citation dictionaries
        """
        citations = []

        for ranked in ranked_results:
            result = ranked.result

            citation = {
                "citation_id": f"CIT-{ranked.rank:03d}",
                "result_id": result.result_id,
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "section": result.section_title,
                "content_type": result.content_type,
                "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "score": ranked.final_score,
                "rank": ranked.rank
            }

            citations.append(citation)

        return citations
