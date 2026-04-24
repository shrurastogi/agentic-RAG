"""
Hybrid Search implementation combining BM25 and vector semantic search.

This module provides:
- BM25 keyword search
- Vector semantic search
- Hybrid search (weighted combination)
- Metadata filtering (doc_id, section, page_range)
- Multi-modal retrieval (text, tables, figures)
"""

from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from loguru import logger
import weaviate
from weaviate.classes.query import MetadataQuery, Filter
import numpy as np

from config.settings import settings
from src.embeddings.vector_store import VectorStore


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    result_id: str
    content_type: str  # "text", "table", "figure"
    content: str
    score: float
    doc_id: str
    page_num: int
    section_title: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "content_type": self.content_type,
            "content": self.content,
            "score": self.score,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "section_title": self.section_title,
            "metadata": self.metadata
        }


@dataclass
class SearchFilters:
    """Filters for search queries."""
    doc_ids: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    page_min: Optional[int] = None
    page_max: Optional[int] = None
    content_types: Optional[List[str]] = None  # ["text", "table", "figure"]
    contains_statistics: Optional[bool] = None


class HybridSearch:
    """
    Hybrid search combining BM25 keyword search and vector semantic search.

    Features:
    - BM25 keyword matching
    - Vector similarity search
    - Configurable weighting (alpha parameter)
    - Metadata filtering
    - Multi-modal content retrieval
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        alpha: Optional[float] = None,
        top_k_text: Optional[int] = None,
        top_k_tables: Optional[int] = None,
        top_k_figures: Optional[int] = None
    ):
        """
        Initialize hybrid search.

        Args:
            vector_store: VectorStore instance (creates new if None)
            alpha: Balance between BM25 (0) and vector (1), default from settings
            top_k_text: Number of text chunks to retrieve
            top_k_tables: Number of tables to retrieve
            top_k_figures: Number of figures to retrieve
        """
        self.vector_store = vector_store or VectorStore()
        self.alpha = alpha if alpha is not None else settings.HYBRID_ALPHA
        self.top_k_text = top_k_text or settings.TOP_K_RETRIEVAL
        self.top_k_tables = top_k_tables or settings.TOP_K_TABLES
        self.top_k_figures = top_k_figures or settings.TOP_K_FIGURES

        logger.info(
            f"HybridSearch initialized: alpha={self.alpha}, "
            f"top_k(text={self.top_k_text}, tables={self.top_k_tables}, figures={self.top_k_figures})"
        )

    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        include_text: bool = True,
        include_tables: bool = True,
        include_figures: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search across all content types.

        Args:
            query: Search query
            filters: Optional filters
            include_text: Whether to search text chunks
            include_tables: Whether to search tables
            include_figures: Whether to search figures

        Returns:
            List of SearchResult objects, sorted by score
        """
        logger.info(f"Hybrid search: '{query[:50]}...'")

        all_results = []

        # Search text chunks
        if include_text:
            text_results = self._search_collection(
                query=query,
                collection_name=settings.TEXT_CHUNK_CLASS,
                limit=self.top_k_text,
                filters=filters,
                content_type="text"
            )
            all_results.extend(text_results)
            logger.debug(f"Found {len(text_results)} text results")

        # Search tables
        if include_tables:
            table_results = self._search_collection(
                query=query,
                collection_name=settings.TABLE_CLASS,
                limit=self.top_k_tables,
                filters=filters,
                content_type="table"
            )
            all_results.extend(table_results)
            logger.debug(f"Found {len(table_results)} table results")

        # Search figures
        if include_figures:
            figure_results = self._search_collection(
                query=query,
                collection_name=settings.FIGURE_CLASS,
                limit=self.top_k_figures,
                filters=filters,
                content_type="figure"
            )
            all_results.extend(figure_results)
            logger.debug(f"Found {len(figure_results)} figure results")

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Retrieved {len(all_results)} total results")

        return all_results

    def _search_collection(
        self,
        query: str,
        collection_name: str,
        limit: int,
        filters: Optional[SearchFilters],
        content_type: str
    ) -> List[SearchResult]:
        """
        Search a specific Weaviate collection using hybrid search.

        Args:
            query: Search query
            collection_name: Weaviate collection name
            limit: Maximum results
            filters: Search filters
            content_type: Type of content ("text", "table", "figure")

        Returns:
            List of SearchResult objects
        """
        try:
            collection = self.vector_store.client.collections.get(collection_name)

            # Generate query embedding
            query_embedding = self.vector_store.generate_embeddings([query])[0]

            # Build filter conditions
            where_filter = self._build_where_filter(filters, content_type)

            # Perform hybrid search
            # Weaviate's hybrid search combines BM25 and vector search
            response = collection.query.hybrid(
                query=query,
                vector=query_embedding.tolist(),
                alpha=self.alpha,  # 0 = pure BM25, 1 = pure vector
                limit=limit,
                return_metadata=MetadataQuery(score=True, distance=True),
                where=where_filter
            )

            # Convert to SearchResult objects
            results = []
            for obj in response.objects:
                result = self._convert_to_search_result(
                    obj,
                    content_type,
                    collection_name
                )
                if result:
                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching {collection_name}: {e}")
            return []

    def _build_where_filter(
        self,
        filters: Optional[SearchFilters],
        content_type: str
    ) -> Optional[Filter]:
        """
        Build Weaviate where filter from SearchFilters.

        Args:
            filters: Search filters
            content_type: Content type being searched

        Returns:
            Weaviate Filter or None
        """
        if not filters:
            return None

        conditions = []

        # Document ID filter
        if filters.doc_ids:
            if len(filters.doc_ids) == 1:
                conditions.append(
                    Filter.by_property("doc_id").equal(filters.doc_ids[0])
                )
            else:
                # Multiple doc_ids - use OR
                doc_filters = [
                    Filter.by_property("doc_id").equal(doc_id)
                    for doc_id in filters.doc_ids
                ]
                conditions.append(Filter.any_of(doc_filters))

        # Page range filter
        if filters.page_min is not None:
            conditions.append(
                Filter.by_property("page_num").greater_or_equal(filters.page_min)
            )

        if filters.page_max is not None:
            conditions.append(
                Filter.by_property("page_num").less_or_equal(filters.page_max)
            )

        # Section filter (text chunks only)
        if content_type == "text" and filters.sections:
            if len(filters.sections) == 1:
                conditions.append(
                    Filter.by_property("section_title").equal(filters.sections[0])
                )
            else:
                section_filters = [
                    Filter.by_property("section_title").equal(section)
                    for section in filters.sections
                ]
                conditions.append(Filter.any_of(section_filters))

        # Statistical content filter (text chunks only)
        if content_type == "text" and filters.contains_statistics is not None:
            conditions.append(
                Filter.by_property("contains_statistics").equal(filters.contains_statistics)
            )

        # Combine conditions with AND
        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return Filter.all_of(conditions)

    def _convert_to_search_result(
        self,
        obj,
        content_type: str,
        collection_name: str
    ) -> Optional[SearchResult]:
        """
        Convert Weaviate object to SearchResult.

        Args:
            obj: Weaviate object
            content_type: Type of content
            collection_name: Collection name

        Returns:
            SearchResult or None
        """
        try:
            props = obj.properties

            # Extract common fields
            doc_id = props.get("doc_id", "")
            page_num = props.get("page_num", 0)

            # Get score (Weaviate hybrid search provides a score)
            score = obj.metadata.score if hasattr(obj.metadata, 'score') else 0.0

            # Content-specific fields
            if content_type == "text":
                result_id = props.get("chunk_id", "")
                content = props.get("text", "")
                section_title = props.get("section_title")
                metadata = {
                    "chunk_index": props.get("chunk_index", 0),
                    "token_count": props.get("token_count", 0),
                    "contains_statistics": props.get("contains_statistics", False)
                }

            elif content_type == "table":
                result_id = props.get("table_id", "")
                # For tables, use summary for display, but include markdown in metadata
                content = props.get("summary", "")
                section_title = None
                metadata = {
                    "markdown": props.get("markdown", ""),
                    "json_structure": props.get("json_structure", ""),
                    "num_rows": props.get("num_rows", 0),
                    "num_cols": props.get("num_cols", 0)
                }

            elif content_type == "figure":
                result_id = props.get("figure_id", "")
                content = props.get("description", "")
                section_title = None
                metadata = {
                    "image_path": props.get("image_path", ""),
                    "figure_type": props.get("figure_type", ""),
                    "ocr_text": props.get("ocr_text", "")
                }
            else:
                return None

            return SearchResult(
                result_id=result_id,
                content_type=content_type,
                content=content,
                score=score,
                doc_id=doc_id,
                page_num=page_num,
                section_title=section_title,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error converting search result: {e}")
            return None

    def search_text_only(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search text chunks only.

        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum results (default: top_k_text)

        Returns:
            List of SearchResult objects
        """
        original_limit = self.top_k_text
        if limit:
            self.top_k_text = limit

        results = self.search(
            query=query,
            filters=filters,
            include_text=True,
            include_tables=False,
            include_figures=False
        )

        self.top_k_text = original_limit
        return results

    def search_tables_only(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search tables only.

        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum results (default: top_k_tables)

        Returns:
            List of SearchResult objects
        """
        original_limit = self.top_k_tables
        if limit:
            self.top_k_tables = limit

        results = self.search(
            query=query,
            filters=filters,
            include_text=False,
            include_tables=True,
            include_figures=False
        )

        self.top_k_tables = original_limit
        return results

    def search_figures_only(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        limit: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search figures only.

        Args:
            query: Search query
            filters: Optional filters
            limit: Maximum results (default: top_k_figures)

        Returns:
            List of SearchResult objects
        """
        original_limit = self.top_k_figures
        if limit:
            self.top_k_figures = limit

        results = self.search(
            query=query,
            filters=filters,
            include_text=False,
            include_tables=False,
            include_figures=True
        )

        self.top_k_figures = original_limit
        return results
