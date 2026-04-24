# Agentic RAG System for Medical Documents

A multi-modal RAG system designed for processing Clinical Study Reports (CSRs) and other medical research documents. The system handles complex PDFs containing tables, graphs, and structured data with full citation tracking and multi-document reasoning capabilities.

## Features

- **Multi-Modal Processing**: Extracts and processes text, tables, and figures separately
- **Semantic Chunking**: Section-aware chunking that preserves context
- **Hybrid Search**: Combines BM25 keyword search with semantic vector search
- **Citation Tracking**: Full provenance tracking with PDF location highlighting
- **Medical Domain Optimized**: Preserves statistical data and maintains precision
- **Agentic Workflows**: LangGraph-based multi-step reasoning
- **Open Source Stack**: Uses Llama 3.1 via Ollama, Weaviate, and BGE embeddings

## Architecture

```
PDF → Content Separation → Parallel Processing → Unified Vector Store
         ↓                      ↓
    Text | Tables | Figures → Specialized Extraction → Weaviate
```

## Tech Stack

- **PDF Processing**: PyMuPDF, Unstructured, Table-Transformer, Camelot
- **Vision & OCR**: Llama 3.2 Vision 11B, EasyOCR
- **Embeddings**: BAAI/bge-large-en-v1.5 (1024 dims)
- **Vector DB**: Weaviate (hybrid search)
- **LLM**: Llama 3.1 8B/70B via Ollama
- **Framework**: LangGraph for agentic workflows

## Current Status: Phase 1 Complete ✓

**Phase 1: Foundation & Basic Processing**
- ✓ [src/ingestion/pdf_parser.py](src/ingestion/pdf_parser.py) - PDF parsing with bounding boxes
- ✓ [src/ingestion/text_processor.py](src/ingestion/text_processor.py) - Semantic chunking
- ✓ [src/embeddings/vector_store.py](src/embeddings/vector_store.py) - Weaviate integration
- ✓ [config/settings.py](config/settings.py) - Configuration management
- ✓ Basic pipeline: PDF → chunks → vector store

## Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Docker** (for Weaviate)
3. **Ollama** (for LLM inference)

### Installation

1. **Clone and install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Start Weaviate:**

```bash
docker-compose up -d
```

3. **Install Ollama and pull models:**

```bash
# Install Ollama from https://ollama.ai

# Pull models
ollama pull llama3.1:8b
ollama pull llama3.2-vision:11b
```

4. **Configure environment:**

```bash
cp .env.example .env
# Edit .env with your settings
```

### Testing Phase 1 Pipeline

```bash
# Test with your own PDF
python scripts/test_phase1_pipeline.py data/raw/your_document.pdf
```

**Expected output:**
- PDF parsing with page count and content region detection
- Text chunking with section awareness
- Weaviate insertion and search verification

## Project Structure

```
agentic-rag/
├── config/
│   └── settings.py              # ✓ Pydantic configuration
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py        # ✓ PyMuPDF extraction
│   │   └── text_processor.py    # ✓ Semantic chunking
│   ├── embeddings/
│   │   └── vector_store.py      # ✓ Weaviate client
│   ├── retrieval/               # TODO: Phase 3
│   ├── agents/                  # TODO: Phase 4
│   ├── llm/                     # TODO
│   ├── citation/                # TODO: Phase 4
│   └── utils/                   # TODO: Phase 5
├── api/                         # TODO: Phase 5
├── cli/                         # TODO: Phase 5
├── data/
│   ├── raw/                     # Place PDF files here
│   ├── processed/               # Extraction artifacts
│   └── cache/                   # Query cache
├── tests/                       # TODO: Phase 6
├── scripts/
│   └── test_phase1_pipeline.py  # ✓ Phase 1 test
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Development Roadmap

### ✓ Phase 1: Foundation (Complete)
- PDF parsing, text chunking, vector store

### Phase 2: Advanced Extraction (Next)
- Table extraction (table-transformer + camelot)
- Figure processing (vision model descriptions)
- Document orchestration pipeline

### Phase 3: Retrieval & Reranking
- Hybrid search implementation
- Query processing with intent classification
- Cross-encoder reranking

### Phase 4: Agentic Workflows & Citations
- LangGraph state machine
- Multi-document reasoning
- Full citation tracking system

### Phase 5: Optimization & Interfaces
- Multi-level caching
- FastAPI REST API
- CLI interface

### Phase 6: Testing & Documentation
- Unit and integration tests
- Deployment documentation
- Performance benchmarks

## Configuration

Key settings in [config/settings.py](config/settings.py):

- **Chunking**: 512 tokens with 100 token overlap
- **Retrieval**: Top 15 initial, rerank to top 8
- **Embeddings**: BGE-large-en-v1.5 (1024 dimensions)
- **LLM Temperature**: 0.1 (low to reduce hallucinations)

All settings can be overridden via environment variables in `.env`.

## Next Steps

Follow the plan in [PLAN.md](PLAN.md):

1. **Phase 2**: Implement table and figure extraction
2. **Phase 3**: Build retrieval pipeline with hybrid search
3. **Phase 4**: Create agentic workflows with citations
4. **Phase 5**: Add API and CLI interfaces
5. **Phase 6**: Comprehensive testing and documentation

## Requirements

See [requirements.txt](requirements.txt) for full dependency list. Key dependencies:

- `pymupdf==1.23.8` - Fast PDF parsing
- `weaviate-client==3.26.1` - Vector database
- `sentence-transformers==2.2.2` - Embeddings
- `langchain==0.1.0` + `langgraph==0.0.38` - LLM framework
- `fastapi==0.109.0` - API framework

## Hardware Recommendations

- **Minimum**: 8-core CPU, 32GB RAM, 500GB SSD
- **Recommended**: 16-core CPU, 64GB RAM, 1TB NVMe, RTX 4090 24GB

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
