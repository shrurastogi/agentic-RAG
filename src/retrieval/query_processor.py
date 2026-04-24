"""
Query Processor for understanding and classifying queries.

This module provides:
- Query intent classification (factual, comparative, table_query, trend_analysis)
- Filter extraction from natural language
- Query decomposition for complex questions
- Query expansion and reformulation
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger

from .hybrid_search import SearchFilters


class QueryIntent(Enum):
    """Query intent types."""
    FACTUAL = "factual"  # "What is X?", "Define Y"
    COMPARATIVE = "comparative"  # "Compare A and B", "Differences between X and Y"
    TABLE_QUERY = "table_query"  # "Show me the table", "What are the statistics"
    TREND_ANALYSIS = "trend_analysis"  # "What's the trend", "How did X change"
    MULTI_DOC = "multi_document"  # "Across all studies", "In all documents"
    STATISTICAL = "statistical"  # "What's the p-value", "Show confidence intervals"


@dataclass
class ProcessedQuery:
    """Represents a processed query with metadata."""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    filters: SearchFilters
    sub_queries: List[str]
    keywords: List[str]
    focus_content_types: List[str]  # ["text", "table", "figure"]
    metadata: Dict


class QueryProcessor:
    """
    Process and understand queries for better retrieval.

    Features:
    - Intent classification
    - Filter extraction (doc names, sections, pages)
    - Query decomposition
    - Keyword extraction
    """

    def __init__(self):
        """Initialize query processor."""
        # Patterns for intent classification
        self.intent_patterns = {
            QueryIntent.COMPARATIVE: [
                r'\b(compare|comparison|versus|vs|difference|differ)\b',
                r'\b(better|worse|higher|lower)\s+than\b',
                r'\b(both|either)\b.*\b(and|or)\b'
            ],
            QueryIntent.TABLE_QUERY: [
                r'\b(table|tables|show.*table|statistics|data)\b',
                r'\b(rows|columns|values)\b',
                r'\b(list|enumerate)\b.*\b(all|values)\b'
            ],
            QueryIntent.TREND_ANALYSIS: [
                r'\b(trend|over time|progression|evolution|change)\b',
                r'\b(increase|decrease|grow|decline)\b',
                r'\b(timeline|history|temporal)\b'
            ],
            QueryIntent.MULTI_DOC: [
                r'\b(all (studies|documents|papers|reports))\b',
                r'\b(across|throughout)\b.*\b(all|multiple)\b',
                r'\b(compare.*studies|studies.*compare)\b'
            ],
            QueryIntent.STATISTICAL: [
                r'\b(p-value|confidence interval|CI|significance|correlation)\b',
                r'\b(mean|median|standard deviation|variance)\b',
                r'\b(odds ratio|hazard ratio|relative risk)\b',
                r'\b(statistical|statistically)\b'
            ]
        }

        # Document name patterns
        self.doc_patterns = [
            r'\b(in|from|about)\s+(?:document|doc|study|paper|report)\s+["\']?([^"\']+)["\']?',
            r'\b(CSR|study|trial)\s+(?:ID|number|#)?\s*[:\-]?\s*([A-Z0-9\-]+)',
        ]

        # Section patterns
        self.section_patterns = [
            r'\b(in|from|under)\s+(?:section|chapter)\s+["\']?([^"\']+)["\']?',
            r'\b(introduction|methods|results|discussion|conclusion|abstract)\b'
        ]

        # Page patterns
        self.page_patterns = [
            r'\bon\s+page\s+(\d+)',
            r'\bpage[s]?\s+(\d+)(?:\s*-\s*(\d+))?',
        ]

        logger.info("QueryProcessor initialized")

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query and extract metadata.

        Args:
            query: User query

        Returns:
            ProcessedQuery object
        """
        logger.debug(f"Processing query: '{query}'")

        # Clean query
        cleaned_query = self._clean_query(query)

        # Classify intent
        intent = self._classify_intent(cleaned_query)

        # Extract filters
        filters = self._extract_filters(cleaned_query)

        # Extract keywords
        keywords = self._extract_keywords(cleaned_query)

        # Decompose complex queries
        sub_queries = self._decompose_query(cleaned_query, intent)

        # Determine focus content types
        focus_types = self._determine_content_types(cleaned_query, intent)

        processed = ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            filters=filters,
            sub_queries=sub_queries,
            keywords=keywords,
            focus_content_types=focus_types,
            metadata={
                "has_filters": filters.doc_ids or filters.sections or filters.page_min,
                "is_complex": len(sub_queries) > 1,
                "requires_statistics": intent == QueryIntent.STATISTICAL
            }
        )

        logger.debug(
            f"Processed query - Intent: {intent.value}, "
            f"Focus: {focus_types}, Sub-queries: {len(sub_queries)}"
        )

        return processed

    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize query.

        Args:
            query: Raw query

        Returns:
            Cleaned query
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Remove trailing question marks (we'll add context later)
        query = query.rstrip('?')

        return query

    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent.

        Args:
            query: Cleaned query

        Returns:
            QueryIntent
        """
        query_lower = query.lower()

        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return intent

        # Default to factual
        return QueryIntent.FACTUAL

    def _extract_filters(self, query: str) -> SearchFilters:
        """
        Extract filters from query.

        Args:
            query: Cleaned query

        Returns:
            SearchFilters object
        """
        filters = SearchFilters()

        # Extract document names/IDs
        doc_ids = []
        for pattern in self.doc_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Match is tuple, take the last group (the doc ID/name)
                doc_id = match[-1] if isinstance(match, tuple) else match
                doc_ids.append(doc_id.strip())

        if doc_ids:
            filters.doc_ids = doc_ids

        # Extract sections
        sections = []
        for pattern in self.section_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                section = match[-1] if isinstance(match, tuple) else match
                sections.append(section.strip().title())

        if sections:
            filters.sections = sections

        # Extract page ranges
        for pattern in self.page_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    # Single page
                    page = int(groups[0])
                    filters.page_min = page
                    filters.page_max = page
                elif len(groups) >= 2 and groups[1]:
                    # Page range
                    filters.page_min = int(groups[0])
                    filters.page_max = int(groups[1])
                else:
                    # Just minimum
                    filters.page_min = int(groups[0])

        # Detect statistical content requirement
        query_lower = query.lower()
        if any(term in query_lower for term in ['statistics', 'statistical', 'p-value', 'confidence']):
            filters.contains_statistics = True

        return filters

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.

        Args:
            query: Cleaned query

        Returns:
            List of keywords
        """
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'what', 'which', 'who',
            'when', 'where', 'why', 'how'
        }

        # Extract words
        words = re.findall(r'\b\w+\b', query.lower())

        # Filter stopwords and short words
        keywords = [
            word for word in words
            if word not in stopwords and len(word) > 2
        ]

        return keywords

    def _decompose_query(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[str]:
        """
        Decompose complex queries into sub-queries.

        Args:
            query: Cleaned query
            intent: Query intent

        Returns:
            List of sub-queries
        """
        sub_queries = [query]  # Always include original

        # Decompose comparative queries
        if intent == QueryIntent.COMPARATIVE:
            # Try to split "compare A and B" into separate lookups
            compare_patterns = [
                r'compare\s+([^and]+)\s+and\s+(.+)',
                r'difference\s+between\s+([^and]+)\s+and\s+(.+)',
                r'([^vs]+)\s+vs\.?\s+(.+)'
            ]

            for pattern in compare_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    entity_a = match.group(1).strip()
                    entity_b = match.group(2).strip()

                    sub_queries.append(f"information about {entity_a}")
                    sub_queries.append(f"information about {entity_b}")
                    break

        # Decompose multi-document queries
        elif intent == QueryIntent.MULTI_DOC:
            # Add sub-query without multi-doc context
            simplified = re.sub(
                r'\b(all|across|throughout)\s+(studies|documents|papers)\b',
                '',
                query,
                flags=re.IGNORECASE
            ).strip()

            if simplified and simplified != query:
                sub_queries.append(simplified)

        return sub_queries

    def _determine_content_types(
        self,
        query: str,
        intent: QueryIntent
    ) -> List[str]:
        """
        Determine which content types to focus on.

        Args:
            query: Cleaned query
            intent: Query intent

        Returns:
            List of content types to prioritize
        """
        query_lower = query.lower()

        # Table-specific queries
        if intent == QueryIntent.TABLE_QUERY or any(
            term in query_lower for term in ['table', 'statistics', 'data', 'rows', 'columns']
        ):
            return ["table", "text"]

        # Figure/trend queries
        if intent == QueryIntent.TREND_ANALYSIS or any(
            term in query_lower for term in ['figure', 'graph', 'chart', 'plot', 'trend', 'visualization']
        ):
            return ["figure", "text"]

        # Statistical queries - prioritize text with statistics
        if intent == QueryIntent.STATISTICAL:
            return ["text", "table"]

        # Default: all types with text first
        return ["text", "table", "figure"]

    def expand_query(self, query: str, domain: str = "medical") -> List[str]:
        """
        Expand query with synonyms and related terms.

        Args:
            query: Original query
            domain: Domain for expansion (default: medical)

        Returns:
            List of expanded queries
        """
        # Medical domain synonym expansion
        medical_synonyms = {
            "efficacy": ["effectiveness", "therapeutic effect", "response rate"],
            "safety": ["adverse events", "side effects", "tolerability"],
            "adverse event": ["side effect", "adverse reaction", "AE"],
            "endpoint": ["outcome", "measure", "objective"],
            "patient": ["subject", "participant"],
            "treatment": ["therapy", "intervention", "regimen"],
            "study": ["trial", "research", "investigation"]
        }

        query_lower = query.lower()
        expanded = [query]

        # Add expansions for matched terms
        for term, synonyms in medical_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)

        return expanded
