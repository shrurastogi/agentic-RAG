# Plan: Optimized RAG System for Medical Documents

## Context
Building an agentic RAG system to process medical research documents (Clinical Study Reports/CSRs) containing complex PDFs with tables, graphs, and structured data. The system needs to handle small scale (< 1000 documents, < 100 queries/day) with all key capabilities: table extraction, graph/figure understanding, multi-document reasoning, and citation tracking.

## Requirements
- **LLM Provider**: Open source (Llama 3.1 8B/70B via Ollama)
- **Vector Database**: Weaviate (hybrid search with BM25 + semantic)
- **Scale**: Small (< 1000 docs, < 100 queries/day) - allows for aggressive caching
- **Critical Features**: 
  - Table extraction and semantic querying
  - Graph/figure understanding via vision models
  - Multi-document comparative reasoning
  - Full citation and provenance tracking

## Architecture Overview

### Multi-Modal Document Processing Pipeline

```
PDF → Content Separation → Parallel Processing → Unified Vector Store
         ↓                      ↓
    Text | Tables | Figures → Specialized Extraction → Weaviate
```

**Key Design Principle**: Treat text, tables, and figures as separate content types with specialized processing, then unify in retrieval.

### Core Technology Stack

**Document Processing**:
- `pymupdf` (PyMuPDF) - Fast PDF parsing with layout preservation & bounding boxes for citations
- `unstructured` - Layout analysis to classify content regions
- `table-transformer` (Microsoft TATR) - State-of-art table structure recognition
- `camelot-py` - Fallback for well-formed tables
- `Llama 3.2 Vision 11B` - Vision model for chart/graph understanding
- `easyocr` - OCR for text elements in figures

**Embeddings & Retrieval**:
- `BAAI/bge-large-en-v1.5` - Embedding model (1024 dims, excellent for long documents)
- `BAAI/bge-reranker-base` - Cross-encoder for reranking
- `Weaviate` - Vector database with hybrid search (BM25 + semantic)

**LLM & Agents**:
- `Llama 3.1 8B Instruct` via Ollama - Main LLM (or 70B quantized for complex queries)
- `LangGraph` - State machine for agentic workflows with explicit control
- `LangChain` - Supporting utilities

## Implementation Plan

### Phase 1: Foundation & Basic Processing (Week 1-2)

**Setup Infrastructure**:
- Install dependencies via [requirements.txt](requirements.txt)
- Setup Weaviate via [docker-compose.yml](docker-compose.yml)
- Configure Ollama with Llama 3.1 8B

**Implement Core Pipeline**:
1. Create [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py):
   - Extract text with PyMuPDF preserving page numbers and bounding boxes
   - Detect content regions (text blocks, tables, figures)

2. Create [src/ingestion/text_processor.py](src/ingestion/text_processor.py):
   - Semantic chunking with section awareness (512 tokens, 100 overlap)
   - Preserve section boundaries for CSR structure
   - Extract metadata: doc_id, page_num, section_title, chunk_index

3. Create [src/embeddings/vector_store.py](src/embeddings/vector_store.py):
   - Define Weaviate schema with 3 classes: TextChunk, Table, Figure
   - Implement embedding generation using BGE
   - Batch insert with metadata

4. Create [config/settings.py](config/settings.py):
   - Pydantic-based configuration management
   - Environment variables for all tunable parameters

**Deliverable**: Working pipeline from PDF → chunks → vector store with basic search

### Phase 2: Advanced Extraction (Week 3-4)

**Table Processing**:
1. Create [src/ingestion/table_extractor.py](src/ingestion/table_extractor.py):
   - Integrate table-transformer for detection & structure recognition
   - Implement camelot-py as fallback
   - Convert tables to: (a) Markdown, (b) JSON structure, (c) LLM-generated summary
   - Store all three representations - markdown for context, JSON for filtering, summary for semantic search

**Figure Processing**:
2. Create [src/ingestion/figure_processor.py](src/ingestion/figure_processor.py):
   - Extract figures as images using PyMuPDF
   - Setup Llama 3.2 Vision to generate detailed descriptions (chart type, axes, trends, key values)
   - Apply OCR for text-heavy figures
   - Store description + image path in vector DB

**Document Orchestration**:
3. Create [src/ingestion/document_loader.py](src/ingestion/document_loader.py):
   - Orchestrate entire multi-modal extraction pipeline
   - Parallel processing of text, tables, figures
   - Error handling and progress tracking
   - Cache processed artifacts to avoid re-extraction

**Deliverable**: Multi-modal content extraction with tables and figures fully searchable

### Phase 3: Retrieval & Reranking (Week 5)

**Hybrid Search**:
1. Create [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py):
   - Implement BM25 + vector hybrid search (alpha=0.5)
   - Add metadata filtering (doc_id, section, page_range)
   - Retrieve: top 10-15 text chunks, top 3-5 tables, top 2-3 figures

**Query Processing**:
2. Create [src/retrieval/query_processor.py](src/retrieval/query_processor.py):
   - LLM-based query understanding: classify intent (factual/comparative/table_query/trend_analysis)
   - Extract filters from natural language (document names, sections)
   - Query decomposition for complex questions

**Reranking**:
3. Create [src/retrieval/reranker.py](src/retrieval/reranker.py):
   - Cross-encoder reranking to get top 5-8 most relevant chunks
   - Context assembly: group by document, preserve metadata for citations

**Deliverable**: Production-ready retrieval with query preprocessing and relevance optimization

### Phase 4: Agentic Workflows & Citations (Week 6-7)

**Agent Framework**:
1. Create [src/agents/graph_builder.py](src/agents/graph_builder.py):
   - Define LangGraph state machine with routing logic:
     ```
     Query Analysis → Route by Intent → [Single Doc | Multi Doc | Table Focus | Figure Focus]
                                              ↓
                                        Retrieve → Synthesize → Cite → Verify
     ```
   - Implement state management for multi-step reasoning

2. Create [src/agents/tools.py](src/agents/tools.py):
   - `search_documents`: Vector + hybrid search with filters
   - `retrieve_table`: Get specific table with structured data
   - `retrieve_figure`: Get figure description + image
   - `compare_across_docs`: Parallel retrieval for comparative queries
   - `extract_statistics`: Pull numerical data
   - `verify_citation`: Check claimed info against sources

3. Create [src/agents/workflows.py](src/agents/workflows.py):
   - Multi-document comparative workflow: decompose → parallel retrieve → synthesize with citations
   - Table-focused workflow: identify relevant tables → extract structured data → summarize

**Citation System**:
4. Create [src/citation/citation_tracker.py](src/citation/citation_tracker.py):
   - Assign unique citation IDs during retrieval: [CIT-001], [CIT-002]
   - Store citation database: {citation_id → {doc_id, title, page_num, section, snippet}}
   - Post-process LLM output to expand inline citations

5. Create [src/citation/provenance.py](src/citation/provenance.py):
   - Verification step: check claims against source chunks
   - Source highlighting using bounding boxes from PyMuPDF extraction
   - "Show Source" functionality to retrieve exact PDF page

**Deliverable**: Agentic multi-document reasoning with full citation tracking

### Phase 5: Optimization & Interfaces (Week 8)

**Caching**:
1. Create [src/utils/cache.py](src/utils/cache.py):
   - Embedding cache (one-time computation)
   - Query result cache with TTL (24 hours)
   - LLM response cache (7 days)
   - Use `diskcache` for simplicity at this scale

**API**:
2. Create [api/main.py](api/main.py) and [api/routes.py](api/routes.py):
   - FastAPI endpoints: `/query`, `/upload`, `/status`, `/citation/{id}`
   - Pydantic models for requests/responses
   - Error handling and logging

**CLI**:
3. Create [cli/main.py](cli/main.py):
   - Interactive query interface with rich output
   - Document upload command
   - System health checks

**Prompts**:
4. Create [config/prompts.yaml](config/prompts.yaml):
   - Medical-domain system prompts emphasizing accuracy, citations, and precision
   - Query decomposition prompts
   - Citation generation prompts
   - Temperature: 0.1-0.2 for reduced hallucinations

**Deliverable**: Production-ready system with API, CLI, and optimizations

### Phase 6: Testing & Documentation (Week 9-10)

**Testing**:
1. Expand [tests/](tests/) with:
   - Unit tests for each module (>80% coverage)
   - Integration tests: PDF → retrieval → answer generation
   - Quality evaluation: NDCG@K, MRR, Precision@K on test Q&A set
   - Fixtures: 2-3 sample CSRs (anonymized) with gold standard Q&A pairs

**Documentation**:
2. Create comprehensive docs:
   - [README.md](README.md): Setup, architecture overview, usage
   - API documentation (auto-generated via FastAPI/Swagger)
   - Architecture documentation with diagrams
   - Deployment guide with Docker

**Deployment**:
3. Create deployment artifacts:
   - [Dockerfile](Dockerfile) for application
   - [docker-compose.yml](docker-compose.yml) with Weaviate + app
   - Environment setup scripts

**Deliverable**: Production-ready, documented, tested system

## Project Structure

```
agentic-rag/
├── config/
│   ├── settings.py              # Pydantic configuration
│   ├── prompts.yaml             # Domain-specific prompts
│   └── weaviate_schema.json     # Vector DB schema (3 classes)
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PyMuPDF extraction
│   │   ├── text_processor.py    # Semantic chunking
│   │   ├── table_extractor.py   # Table-transformer + camelot
│   │   ├── figure_processor.py  # Vision model processing
│   │   └── document_loader.py   # Orchestration
│   ├── embeddings/
│   │   ├── embedding_model.py   # BGE embeddings
│   │   └── vector_store.py      # Weaviate client
│   ├── retrieval/
│   │   ├── hybrid_search.py     # BM25 + vector
│   │   ├── reranker.py          # Cross-encoder
│   │   └── query_processor.py   # Intent classification
│   ├── agents/
│   │   ├── graph_builder.py     # LangGraph state machine
│   │   ├── tools.py             # Agent tools
│   │   └── workflows.py         # Multi-doc reasoning
│   ├── llm/
│   │   ├── model_manager.py     # Ollama integration
│   │   └── prompt_builder.py    # Dynamic prompts
│   ├── citation/
│   │   ├── citation_tracker.py  # Citation management
│   │   └── provenance.py        # Verification
│   └── utils/
│       └── cache.py             # Multi-level caching
├── api/
│   ├── main.py                  # FastAPI app
│   └── routes.py                # Endpoints
├── cli/
│   └── main.py                  # CLI interface
├── data/
│   ├── raw/                     # Original PDFs
│   ├── processed/               # Extracted artifacts
│   └── cache/                   # Query cache
├── tests/
│   └── fixtures/                # Test CSRs + Q&A pairs
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Critical Files to Implement First

1. [src/ingestion/document_loader.py](src/ingestion/document_loader.py) - Orchestrates entire pipeline
2. [src/ingestion/table_extractor.py](src/ingestion/table_extractor.py) - Critical for CSR tables
3. [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py) - Core retrieval engine
4. [src/agents/graph_builder.py](src/agents/graph_builder.py) - Multi-document reasoning
5. [config/settings.py](config/settings.py) - Central configuration

## Key Design Decisions

**Why PyMuPDF over pdfplumber?**
- 3-5x faster, better layout preservation, provides bounding boxes for exact citation locations

**Why table-transformer over regex/heuristics?**
- State-of-art deep learning model specifically trained for complex table structure recognition - essential for CSR tables with merged cells and nested headers

**Why separate storage for tables vs. text?**
- Tables need different treatment: store markdown for context, JSON for filtering, summary for semantic search
- Enables specialized querying ("show me all efficacy tables from Phase III studies")

**Why LangGraph over CrewAI?**
- Medical domain requires deterministic, auditable workflows with explicit state management
- Easier debugging when citations or reasoning fail

**Why vision model for figures?**
- OCR alone misses context - vision models understand chart semantics (trends, comparisons)
- Critical for safety/efficacy graphs in CSRs

## Medical Domain Optimizations

**Chunking Strategy**:
- 512 tokens with 100 overlap, but preserve section boundaries
- Keep statistical result sections intact
- Never split tables or figure captions

**Metadata for CSRs**:
- Document-level: study_id, phase, therapeutic_area, indication, sponsor
- Chunk-level: section_hierarchy, contains_table/figure, statistical_content flags
- Enables filtering: "Find Phase III oncology safety data"

**Prompt Engineering**:
- Low temperature (0.1-0.2) to reduce hallucinations
- Explicit instructions to preserve exact statistics with confidence intervals
- "I don't know" responses when information not in sources
- Never make clinical recommendations

**Citation Requirements**:
- Inline markers: "Response rate was 78% [CIT-001]"
- Expanded: "[CIT-001] → Smith et al., Page 45, Results Section"
- Verification step to check facts against sources
- Highlight exact location in source PDF

## Verification Strategy

**End-to-End Testing**:
1. Upload 3 sample CSRs via CLI: `python cli/main.py upload data/raw/sample_csr.pdf`
2. Verify extraction: Check that tables and figures are in Weaviate
3. Test queries:
   - Simple factual: "What was the primary endpoint?"
   - Table query: "Show me the adverse events table"
   - Multi-doc: "Compare efficacy rates across all three studies"
   - Figure: "What does the Kaplan-Meier curve show?"
4. Verify citations: Each answer should have [CIT-XXX] markers that resolve to correct sources
5. Check provenance: Click "Show Source" → should highlight exact text in PDF

**Quality Metrics**:
- Retrieval: NDCG@10, MRR on test Q&A set
- Answer quality: LLM-as-judge comparing to gold standard answers
- Citation accuracy: Manual verification that citations point to correct content

## Resource Requirements

**Hardware**:
- Minimum: 8-core CPU, 32GB RAM, 500GB SSD (CPU inference, slower)
- Recommended: 16-core CPU, 64GB RAM, 1TB NVMe SSD, NVIDIA RTX 4090 24GB (for Llama 70B + Vision)

**Performance Estimates**:
- Document ingestion: ~5-10 minutes per 100-page CSR
- Query processing: ~3-6 seconds (with caching: ~500ms)
- Small scale allows aggressive caching - most queries will be sub-second after first run

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Poor table extraction | Multi-tool approach (transformer + camelot), manual review pipeline |
| Hallucinations | Low temperature, citation verification, conservative prompts, log all responses |
| Inaccurate citations | Automatic verification step, store exact snippets, highlight in PDF |
| Complex multi-doc queries fail | Explicit query decomposition, structured workflows, fallback to simpler queries |

## Next Steps

After approval, implementation will proceed phase-by-phase with:
1. Setup development environment (Weaviate, Ollama, dependencies)
2. Implement foundation (PDF → chunks → vector store)
3. Add advanced extraction (tables, figures)
4. Build retrieval pipeline
5. Create agentic workflows with citations
6. Add optimization and interfaces
7. Test and document
