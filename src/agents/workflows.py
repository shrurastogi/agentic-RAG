"""
Agentic Workflows for complex reasoning tasks.

Workflows:
- MultiDocumentComparison: Compare information across documents
- TableFocusedWorkflow: Find and extract table data
- StatisticalAnalysisWorkflow: Extract and analyze statistics
- FactualQAWorkflow: Answer factual questions with citations
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger

from .tools import AgentTools, ToolResult
from src.retrieval.query_processor import QueryProcessor, ProcessedQuery, QueryIntent
from src.retrieval.reranker import Reranker


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    success: bool
    answer: str
    sources: List[Dict]
    reasoning_steps: List[str]
    metadata: Dict = field(default_factory=dict)


class BaseWorkflow:
    """Base class for workflows."""

    def __init__(
        self,
        tools: Optional[AgentTools] = None,
        query_processor: Optional[QueryProcessor] = None,
        reranker: Optional[Reranker] = None
    ):
        """
        Initialize workflow.

        Args:
            tools: AgentTools instance
            query_processor: QueryProcessor instance
            reranker: Reranker instance
        """
        self.tools = tools or AgentTools()
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker or Reranker()

        self.reasoning_steps = []

    def add_step(self, step: str):
        """Add a reasoning step."""
        self.reasoning_steps.append(step)
        logger.debug(f"[Step {len(self.reasoning_steps)}] {step}")

    def execute(self, query: str, **kwargs) -> WorkflowResult:
        """Execute workflow. To be implemented by subclasses."""
        raise NotImplementedError


class MultiDocumentComparison(BaseWorkflow):
    """
    Workflow for comparing information across multiple documents.

    Steps:
    1. Process query to extract comparison aspect
    2. Decompose into sub-queries if needed
    3. Search each document in parallel
    4. Extract relevant information
    5. Synthesize comparison with citations
    """

    def execute(
        self,
        query: str,
        doc_ids: Optional[List[str]] = None
    ) -> WorkflowResult:
        """
        Execute multi-document comparison.

        Args:
            query: Comparison query
            doc_ids: Documents to compare (None = all docs)

        Returns:
            WorkflowResult with comparison
        """
        logger.info(f"MultiDocumentComparison: '{query[:50]}...'")
        self.reasoning_steps = []

        # Step 1: Process query
        self.add_step("Processing query to understand comparison aspect")
        processed = self.query_processor.process(query)

        # Step 2: Identify documents to compare
        if not doc_ids:
            doc_ids = processed.filters.doc_ids or []

        if len(doc_ids) < 2:
            return WorkflowResult(
                success=False,
                answer="Need at least 2 documents to compare. Please specify document IDs.",
                sources=[],
                reasoning_steps=self.reasoning_steps,
                metadata={"error": "insufficient_documents"}
            )

        self.add_step(f"Comparing {len(doc_ids)} documents: {', '.join(doc_ids)}")

        # Step 3: Search each document
        self.add_step("Retrieving relevant information from each document")

        comparison_result = self.tools.compare_across_docs(
            query=query,
            doc_ids=doc_ids,
            aspect=processed.intent.value
        )

        if not comparison_result.success:
            return WorkflowResult(
                success=False,
                answer=f"Comparison failed: {comparison_result.message}",
                sources=[],
                reasoning_steps=self.reasoning_steps
            )

        # Step 4: Synthesize comparison
        self.add_step("Synthesizing comparison across documents")

        comparison_data = comparison_result.data
        answer_parts = [f"Comparison of {len(doc_ids)} documents:"]

        sources = []
        for doc_id, doc_data in comparison_data["documents"].items():
            answer_parts.append(f"\n**{doc_id}**:")

            for idx, result in enumerate(doc_data["top_results"][:3], 1):
                answer_parts.append(
                    f"  - {result['content'][:150]}... "
                    f"(Page {result['page_num']}, Score: {result['score']:.2f})"
                )

                sources.append({
                    "doc_id": doc_id,
                    "page_num": result["page_num"],
                    "section": result["section"],
                    "content": result["content"]
                })

        answer = "\n".join(answer_parts)

        self.add_step(f"Comparison complete with {len(sources)} sources")

        return WorkflowResult(
            success=True,
            answer=answer,
            sources=sources,
            reasoning_steps=self.reasoning_steps,
            metadata={
                "num_documents": len(doc_ids),
                "num_sources": len(sources),
                "aspect": comparison_data["aspect"]
            }
        )


class TableFocusedWorkflow(BaseWorkflow):
    """
    Workflow for table-focused queries.

    Steps:
    1. Identify table requirements
    2. Search for relevant tables
    3. Extract structured data
    4. Summarize findings
    """

    def execute(
        self,
        query: str,
        doc_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute table-focused workflow.

        Args:
            query: Table query
            doc_id: Optional document filter

        Returns:
            WorkflowResult with table data
        """
        logger.info(f"TableFocusedWorkflow: '{query[:50]}...'")
        self.reasoning_steps = []

        # Step 1: Process query
        self.add_step("Identifying table requirements from query")
        processed = self.query_processor.process(query)

        # Extract doc_id from filters if not provided
        if not doc_id and processed.filters.doc_ids:
            doc_id = processed.filters.doc_ids[0]

        # Step 2: Retrieve tables
        self.add_step(f"Searching for tables{' in ' + doc_id if doc_id else ''}")

        table_result = self.tools.retrieve_table(
            query=query,
            doc_id=doc_id,
            limit=5
        )

        if not table_result.success or not table_result.data:
            return WorkflowResult(
                success=False,
                answer="No relevant tables found.",
                sources=[],
                reasoning_steps=self.reasoning_steps
            )

        # Step 3: Format table data
        self.add_step(f"Found {len(table_result.data)} relevant tables")

        tables = table_result.data
        answer_parts = [f"Found {len(tables)} relevant tables:"]

        sources = []
        for idx, table in enumerate(tables, 1):
            answer_parts.append(
                f"\n**Table {idx}** (Page {table['page_num']}, "
                f"{table['num_rows']}×{table['num_cols']}):"
            )
            answer_parts.append(f"  Summary: {table['summary']}")

            # Show markdown preview
            if table['markdown']:
                lines = table['markdown'].split('\n')[:5]
                answer_parts.append("  Preview:")
                for line in lines:
                    answer_parts.append(f"    {line}")

            sources.append({
                "table_id": table['table_id'],
                "doc_id": table['doc_id'],
                "page_num": table['page_num'],
                "content": table['summary'],
                "markdown": table['markdown']
            })

        answer = "\n".join(answer_parts)

        self.add_step("Table extraction complete")

        return WorkflowResult(
            success=True,
            answer=answer,
            sources=sources,
            reasoning_steps=self.reasoning_steps,
            metadata={
                "num_tables": len(tables),
                "doc_id": doc_id
            }
        )


class StatisticalAnalysisWorkflow(BaseWorkflow):
    """
    Workflow for extracting and analyzing statistics.

    Steps:
    1. Identify statistical requirements
    2. Search for statistical content
    3. Extract numerical data
    4. Organize and present findings
    """

    def execute(
        self,
        query: str,
        stat_type: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute statistical analysis workflow.

        Args:
            query: Statistical query
            stat_type: Type of statistic (p-value, CI, etc.)

        Returns:
            WorkflowResult with statistics
        """
        logger.info(f"StatisticalAnalysisWorkflow: '{query[:50]}...'")
        self.reasoning_steps = []

        # Step 1: Process query
        self.add_step("Identifying statistical requirements")
        processed = self.query_processor.process(query)

        # Step 2: Extract statistics
        self.add_step(f"Searching for {stat_type or 'all'} statistics")

        stats_result = self.tools.extract_statistics(
            query=query,
            stat_type=stat_type
        )

        if not stats_result.success or not stats_result.data:
            return WorkflowResult(
                success=False,
                answer="No statistical data found.",
                sources=[],
                reasoning_steps=self.reasoning_steps
            )

        # Step 3: Organize statistics
        self.add_step(f"Found statistics in {len(stats_result.data)} sources")

        statistics = stats_result.data
        answer_parts = [f"Statistical findings from {len(statistics)} sources:"]

        sources = []
        for idx, stat_source in enumerate(statistics, 1):
            answer_parts.append(
                f"\n**Source {idx}** ({stat_source['doc_id']}, "
                f"Page {stat_source['page_num']}):"
            )

            for stat in stat_source['statistics']:
                answer_parts.append(f"  - {stat['type']}: {stat['value']}")

            answer_parts.append(f"  Context: {stat_source['context'][:100]}...")

            sources.append({
                "source_id": stat_source['source'],
                "doc_id": stat_source['doc_id'],
                "page_num": stat_source['page_num'],
                "section": stat_source['section'],
                "statistics": stat_source['statistics'],
                "context": stat_source['context']
            })

        answer = "\n".join(answer_parts)

        self.add_step("Statistical analysis complete")

        return WorkflowResult(
            success=True,
            answer=answer,
            sources=sources,
            reasoning_steps=self.reasoning_steps,
            metadata={
                "num_sources": len(statistics),
                "stat_type": stat_type
            }
        )


class FactualQAWorkflow(BaseWorkflow):
    """
    Workflow for answering factual questions with citations.

    Steps:
    1. Process question
    2. Search for relevant information
    3. Rerank by relevance
    4. Extract answer
    5. Add citations
    """

    def execute(
        self,
        query: str,
        top_k: int = 5
    ) -> WorkflowResult:
        """
        Execute factual QA workflow.

        Args:
            query: Factual question
            top_k: Number of top sources to use

        Returns:
            WorkflowResult with answer and citations
        """
        logger.info(f"FactualQAWorkflow: '{query[:50]}...'")
        self.reasoning_steps = []

        # Step 1: Process query
        self.add_step("Processing factual question")
        processed = self.query_processor.process(query)

        # Step 2: Search
        self.add_step("Searching for relevant information")

        search_result = self.tools.search_documents(
            query=query,
            doc_ids=processed.filters.doc_ids,
            sections=processed.filters.sections,
            content_types=processed.focus_content_types,
            limit=15
        )

        if not search_result.success or not search_result.data:
            return WorkflowResult(
                success=False,
                answer="No relevant information found.",
                sources=[],
                reasoning_steps=self.reasoning_steps
            )

        # Step 3: Rerank
        self.add_step(f"Reranking {len(search_result.data)} results")

        ranked = self.reranker.rerank(
            query=query,
            results=search_result.data,
            top_k=top_k
        )

        # Step 4: Assemble answer
        self.add_step(f"Assembling answer from top {len(ranked)} sources")

        # Create answer from top results
        answer_parts = []

        for idx, ranked_result in enumerate(ranked, 1):
            result = ranked_result.result
            citation_id = f"[{idx}]"

            # Add content with citation
            answer_parts.append(
                f"{result.content[:200]}... {citation_id}"
            )

        answer = "\n\n".join(answer_parts)

        # Prepare sources
        sources = []
        for idx, ranked_result in enumerate(ranked, 1):
            result = ranked_result.result
            sources.append({
                "citation_id": idx,
                "doc_id": result.doc_id,
                "page_num": result.page_num,
                "section": result.section_title,
                "content": result.content,
                "score": ranked_result.final_score
            })

        self.add_step(f"Answer complete with {len(sources)} citations")

        return WorkflowResult(
            success=True,
            answer=answer,
            sources=sources,
            reasoning_steps=self.reasoning_steps,
            metadata={
                "num_sources": len(sources),
                "intent": processed.intent.value
            }
        )
