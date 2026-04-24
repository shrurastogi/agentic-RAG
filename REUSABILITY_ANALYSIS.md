# Plan: Making the RAG System Pluggable for Any Document Domain

## Context

You've built a comprehensive Agentic RAG system currently optimized for medical documents (Clinical Study Reports). You want to know if this code can be plugged into other RAG applications for different document types (legal, financial, technical, etc.) without modifications, or what needs to change.

**Why this matters:**
- Code reusability across different domains
- Avoiding duplicate development for each new use case
- Creating a general-purpose RAG framework instead of a medical-specific one
- Reducing maintenance burden across multiple deployments

## Current State Analysis

After thorough exploration of the codebase, here's what I found:

### ✅ What's Already Reusable (90-95%)

**Ingestion Pipeline** (`src/ingestion/`):
- PDF parsing with PyMuPDF: 100% domain-agnostic
- Text chunking: 95% generic (section-aware, configurable)
- Table extraction with Camelot: 100% domain-agnostic
- Figure processing: 95% generic
- Document orchestration: 100% generic

**Retrieval Components** (`src/retrieval/`):
- Hybrid search (BM25 + vector): 100% domain-agnostic
- Reranker with cross-encoder: 100% domain-agnostic
- Context assembly and grouping: 100% generic

**Core Infrastructure**:
- Vector store integration (Weaviate): Domain-agnostic
- Configuration management: Domain-agnostic
- Parallel processing: Domain-agnostic

### ⚠️ What Has Medical Domain Coupling (Needs Adaptation)

**Query Processing** (`src/retrieval/query_processor.py`):
- **Lines 374-407**: Hardcoded medical synonyms dictionary
  - "efficacy" → ["effectiveness", "therapeutic effect", "response rate"]
  - "safety" → ["adverse events", "side effects", "tolerability"]
- **Lines 78-83**: Medical statistical patterns in intent classification
  - "odds ratio", "hazard ratio", "relative risk"

**Agent Tools** (`src/agents/tools.py`):
- **Lines 463-519**: `_extract_stat_patterns()` method extracts medical-specific statistics
  - Hazard ratios, odds ratios (clinical research specific)
  - P-values, confidence intervals (universal but medical-focused)

**Text Processing** (`src/ingestion/text_processor.py`):
- **Lines 303-340**: `_contains_statistics()` method uses medical terminology
  - Not harmful for other domains, just adds unnecessary metadata

**Settings** (`config/settings.py`):
- **Lines 110-113**: Medical-specific flags (defined but mostly unused)
  - `MEDICAL_TERMINOLOGY_MODE`
  - `PRESERVE_STATISTICS`
  - `CONFIDENCE_INTERVAL_PRECISION`

### 🔴 Critical Extensibility Barriers

**Hard-coded Dependencies** (No abstraction layers):
1. **Vector Store**: Tightly coupled to Weaviate only
   - No interface to swap for Pinecone, pgvector, Milvus, FAISS
2. **PDF Parser**: Hardcoded to PyMuPDF
   - No interface to swap for pdfplumber, pypdf, PDFMiner
3. **Embeddings**: Tied to Sentence-Transformers library
   - Cannot easily use OpenAI embeddings, Cohere, or custom providers
4. **LLM**: Defaulted to Ollama
   - No provider abstraction for OpenAI, Claude, etc.

## Recommended Approach

### Phase 1: Domain Decoupling (High Priority)

**Make domain-specific logic configurable and pluggable:**

#### 1.1 Externalize Medical Synonyms
**File**: `src/retrieval/query_processor.py`

Create domain-specific synonym files:
```
config/domains/
  ├── medical_synonyms.json
  ├── legal_synonyms.json
  ├── financial_synonyms.json
  └── technical_synonyms.json
```

Add `DOMAIN` setting to `config/settings.py`:
```python
DOMAIN: str = "medical"  # Options: medical, legal, financial, technical, general
CUSTOM_SYNONYMS_PATH: Optional[str] = None
```

Modify `QueryProcessor.__init__()` to load synonyms based on domain:
```python
def __init__(self, domain: Optional[str] = None):
    self.domain = domain or settings.DOMAIN
    self.synonyms = self._load_domain_synonyms(self.domain)
```

#### 1.2 Make Statistical Pattern Extraction Pluggable
**File**: `src/agents/tools.py`

Create pattern extractor interface:
```python
class PatternExtractor(ABC):
    @abstractmethod
    def extract_patterns(self, text: str, pattern_type: Optional[str]) -> List[Dict]:
        pass
```

Implement domain-specific extractors:
- `MedicalPatternExtractor`: p-values, hazard ratios, odds ratios, CIs
- `LegalPatternExtractor`: clause numbers, dates, party names, penalties
- `FinancialPatternExtractor`: P/E ratios, revenue, margins, growth rates
- `TechnicalPatternExtractor`: function signatures, classes, API patterns

Modify `AgentTools.__init__()` to accept extractor:
```python
def __init__(
    self,
    pattern_extractor: Optional[PatternExtractor] = None,
    ...
):
    self.pattern_extractor = pattern_extractor or self._get_default_extractor()
```

#### 1.3 Make Hardcoded Thresholds Configurable
**File**: `src/ingestion/pdf_parser.py`

Add to `config/settings.py`:
```python
# Document parsing thresholds
HEADER_FONT_SIZE_THRESHOLD: float = 14.0
MIN_TABLE_ROWS: int = 5
TABLE_COLUMN_ALIGNMENT_TOLERANCE: float = 10.0
```

Update `PDFParser.__init__()`:
```python
def __init__(
    self,
    preserve_layout: bool = True,
    header_threshold: Optional[float] = None
):
    self.header_threshold = header_threshold or settings.HEADER_FONT_SIZE_THRESHOLD
```

Change line 236:
```python
# Before:
is_header = avg_font_size > 14  # Hard-coded

# After:
is_header = avg_font_size > self.header_threshold
```

### Phase 2: Abstraction Layer for True Plug-and-Play (Medium Priority)

**Create interfaces for swappable components:**

#### 2.1 Vector Store Abstraction
**New file**: `src/embeddings/vector_db_interface.py`

```python
class VectorDBInterface(ABC):
    @abstractmethod
    def create_schema(self) -> None:
        """Create necessary schemas/collections"""
        pass
    
    @abstractmethod
    def insert_text_chunks(self, chunks: List[TextChunk]) -> None:
        pass
    
    @abstractmethod
    def search_text_chunks(self, query: str, limit: int, filters: Optional[SearchFilters]) -> List[SearchResult]:
        pass
    
    # ... other methods
```

**Implementations**:
- `WeaviateStore(VectorDBInterface)` - existing implementation
- `PineconeStore(VectorDBInterface)` - new optional implementation
- `PgVectorStore(VectorDBInterface)` - new optional implementation

**Update**: `src/retrieval/hybrid_search.py`
- Change `HybridSearch.__init__()` to accept `VectorDBInterface` instead of `VectorStore`
- Add setting: `VECTOR_DB_PROVIDER: str = "weaviate"` (options: weaviate, pinecone, pgvector)

#### 2.2 PDF Parser Abstraction
**New file**: `src/ingestion/pdf_parser_interface.py`

```python
class PDFParserInterface(ABC):
    @abstractmethod
    def parse_pdf(self, pdf_path: Path, doc_id: Optional[str]) -> ParsedDocument:
        pass
```

**Implementations**:
- `PyMuPDFParser(PDFParserInterface)` - existing
- `PdfplumberParser(PDFParserInterface)` - alternative
- `UnstructuredParser(PDFParserInterface)` - alternative

#### 2.3 Embedding Provider Abstraction
**New file**: `src/embeddings/embedding_provider_interface.py`

```python
class EmbeddingProviderInterface(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        pass
```

**Implementations**:
- `LocalEmbeddingProvider(EmbeddingProviderInterface)` - Sentence-Transformers
- `OpenAIEmbeddingProvider(EmbeddingProviderInterface)` - OpenAI API
- `CohereEmbeddingProvider(EmbeddingProviderInterface)` - Cohere API

### Phase 3: Domain Configuration Profiles (Low Priority - Nice to Have)

**Create pre-configured domain profiles:**

**New file**: `config/domains/profiles.yaml`

```yaml
medical:
  synonyms_file: "medical_synonyms.json"
  pattern_extractor: "MedicalPatternExtractor"
  header_threshold: 14.0
  min_table_rows: 5
  chunk_size: 512
  enable_statistics: true

legal:
  synonyms_file: "legal_synonyms.json"
  pattern_extractor: "LegalPatternExtractor"
  header_threshold: 16.0
  min_table_rows: 3
  chunk_size: 768
  enable_statistics: false

financial:
  synonyms_file: "financial_synonyms.json"
  pattern_extractor: "FinancialPatternExtractor"
  header_threshold: 14.0
  min_table_rows: 5
  chunk_size: 512
  enable_statistics: true

technical:
  synonyms_file: "technical_synonyms.json"
  pattern_extractor: "TechnicalPatternExtractor"
  header_threshold: 12.0
  min_table_rows: 10
  chunk_size: 1024
  enable_statistics: false
```

**Usage**:
```python
# Load profile
from config.domains import load_domain_profile
profile = load_domain_profile("legal")

# Initialize components with profile
query_processor = QueryProcessor(synonyms=profile.synonyms)
tools = AgentTools(pattern_extractor=profile.pattern_extractor)
```

## Critical Files to Modify

### High Priority (Domain Decoupling)
1. **src/retrieval/query_processor.py**
   - Lines 374-407: Externalize medical_synonyms dictionary
   - Add domain loading logic

2. **src/agents/tools.py**
   - Lines 463-519: Extract `_extract_stat_patterns()` to pluggable interface
   - Create domain-specific pattern extractors

3. **config/settings.py**
   - Add `DOMAIN` setting
   - Add configurable thresholds (HEADER_FONT_SIZE_THRESHOLD, MIN_TABLE_ROWS)

4. **src/ingestion/pdf_parser.py**
   - Line 236: Make header threshold configurable
   - Line 298: Make table row threshold configurable

### Medium Priority (Abstraction Layers)
5. **src/embeddings/vector_db_interface.py** (NEW)
   - Create abstract interface

6. **src/embeddings/vector_store.py**
   - Refactor to implement `VectorDBInterface`

7. **src/retrieval/hybrid_search.py**
   - Accept `VectorDBInterface` instead of concrete `VectorStore`

### Low Priority (Domain Profiles)
8. **config/domains/** (NEW directory)
   - Create synonym JSON files per domain
   - Create profiles.yaml
   - Create domain loader utility

## Verification Plan

### Test 1: Medical Domain (Baseline - Should Still Work)
```bash
# Should work exactly as before
export DOMAIN=medical
python scripts/test_phase2_pipeline.py data/raw/medical_csr.pdf
```

**Expected**: Extracts medical statistics, uses medical synonyms, works identically to current version.

### Test 2: Legal Domain (New)
```bash
# Test with legal documents
export DOMAIN=legal
python scripts/test_phase2_pipeline.py data/raw/legal_contract.pdf
```

**Expected**: 
- Uses legal synonyms (plaintiff, defendant, clause)
- Extracts legal patterns (dates, penalties, party names)
- No medical-specific processing

### Test 3: Financial Domain (New)
```bash
# Test with financial reports
export DOMAIN=financial
python scripts/test_phase2_pipeline.py data/raw/financial_10k.pdf
```

**Expected**:
- Uses financial synonyms (revenue, equity, margin)
- Extracts financial metrics (P/E, ROI, revenue growth)
- Tables extracted with financial context

### Test 4: Vector Store Abstraction (If Implemented)
```bash
# Test with alternative vector store
export VECTOR_DB_PROVIDER=pinecone
export PINECONE_API_KEY=your_key
export PINECONE_ENVIRONMENT=us-east-1
python scripts/test_full_pipeline.py data/raw/test_doc.pdf
```

**Expected**: System works identically but uses Pinecone instead of Weaviate.

### Test 5: Query Processing Across Domains
```python
# Test query processor with different domains
processor_medical = QueryProcessor(domain="medical")
processed = processor_medical.process("What was the efficacy rate?")
# Should expand: efficacy → [effectiveness, therapeutic effect, response rate]

processor_financial = QueryProcessor(domain="financial")
processed = processor_financial.process("What was the revenue growth?")
# Should expand: revenue → [sales, turnover, income]
```

### Test 6: Pattern Extraction Verification
```python
# Medical patterns
medical_tools = AgentTools(pattern_extractor=MedicalPatternExtractor())
result = medical_tools.extract_statistics("HR=0.72 (95% CI: 0.60-0.85, p<0.001)")
# Should extract: HR, CI, p-value

# Legal patterns
legal_tools = AgentTools(pattern_extractor=LegalPatternExtractor())
result = legal_tools.extract_statistics("Clause 5.3, penalty $50,000, dated Jan 1, 2024")
# Should extract: clause number, penalty amount, date
```

## Success Criteria

✅ **Must Have (Phase 1)**:
- [ ] Can set `DOMAIN=legal` and system uses legal-specific synonyms
- [ ] Pattern extraction works for medical, legal, and financial domains
- [ ] All hardcoded thresholds are configurable via settings
- [ ] Medical documents still process exactly as before (regression test)

✅ **Should Have (Phase 2)**:
- [ ] Can swap vector stores via configuration (Weaviate → Pinecone)
- [ ] Can swap PDF parsers via configuration
- [ ] Each component has clear interface/abstraction

✅ **Nice to Have (Phase 3)**:
- [ ] Pre-configured domain profiles load automatically
- [ ] Zero code changes needed to switch domains (just config)
- [ ] Documentation shows how to add new domain support

## Effort Estimate

- **Phase 1 (Domain Decoupling)**: 6-8 hours
  - Externalize synonyms: 1-2 hours
  - Pattern extractor interface: 3-4 hours  
  - Configurable thresholds: 1 hour
  - Testing: 1-2 hours

- **Phase 2 (Abstraction Layers)**: 8-12 hours
  - Vector DB interface: 4-6 hours
  - PDF parser interface: 2-3 hours
  - Embedding provider interface: 2-3 hours

- **Phase 3 (Domain Profiles)**: 2-3 hours
  - Create profile files: 1 hour
  - Profile loader: 1-2 hours

**Total: 16-23 hours for complete generalization**

## Conclusion

**Current Answer to Your Question:**

> "Can the code be plugged into any RAG application for similar documents, or do we need modifications?"

**Answer: Needs moderate modifications (Phase 1) for true plug-and-play across domains.**

- **Ingestion pipeline (90%)**: Nearly ready - just make 2 thresholds configurable
- **Retrieval/reranking (95%)**: Ready - fully domain-agnostic
- **Query processing (70%)**: Needs synonym externalization
- **Agent tools (60%)**: Needs pattern extractor abstraction
- **Overall reusability: 75%** without changes, **95%** with Phase 1 changes

**Recommended Path**:
1. Implement Phase 1 (6-8 hours) → Gets you to 95% reusability
2. Phase 2 only if you need to swap vector stores/parsers
3. Phase 3 for ultimate ease of use

The system is well-architected overall - just needs domain-specific code extracted into configurable modules.
