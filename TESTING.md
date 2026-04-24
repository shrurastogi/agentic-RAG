# Phase 1 Testing Guide

## Prerequisites Check

✓ Python 3.10+ installed  
✓ Dependencies installed  
⚠️ **Weaviate needs to be started**  
⚠️ **Test PDF needed**  

## Step 1: Start Weaviate

### Option A: Using Docker Desktop (Windows)

1. **Start Docker Desktop** if not already running
2. **Open PowerShell or Command Prompt** and run:
   ```powershell
   cd "c:\Users\sr487403\OneDrive - GSK\Documents\Agentic RAG"
   docker-compose up -d
   ```
3. **Verify Weaviate is running:**
   ```powershell
   curl http://localhost:8080/v1/.well-known/ready
   ```
   Should return: `{"status": "ok"}`

### Option B: Using Docker from WSL/Git Bash

If you have Git Bash or WSL:
```bash
cd "/c/Users/sr487403/OneDrive - GSK/Documents/Agentic RAG"
docker-compose up -d
```

### Troubleshooting

If Docker is not installed:
1. Download from: https://www.docker.com/products/docker-desktop
2. Install and restart your computer
3. Run the commands above

## Step 2: Get a Test PDF

You need a PDF file to test. Options:

### Option A: Use your own PDF
Place any PDF file in `data/raw/` folder. For example:
- Clinical Study Report
- Research paper
- Any multi-page PDF document

### Option B: Download a sample PDF
```powershell
# Example: Download a sample research paper
curl -o data/raw/sample.pdf https://arxiv.org/pdf/1706.03762.pdf
```

## Step 3: Run the Test

```bash
python scripts/test_phase1_pipeline.py data/raw/your_file.pdf
```

Replace `your_file.pdf` with the actual PDF filename.

## Expected Output

The test script will:
1. ✓ Parse the PDF (show page count, text blocks)
2. ✓ Create text chunks (show chunk count)
3. ✓ Connect to Weaviate
4. ✓ Create schema (3 collections: TextChunk, Table, Figure)
5. ✓ Insert chunks with embeddings
6. ✓ Verify insertion (show collection stats)
7. ✓ Test semantic search (3 sample queries)

## What Success Looks Like

```
============================================================
TESTING PHASE 1 PIPELINE
============================================================

[STEP 1] Parsing PDF...
✓ Parsed: Sample Document
  - Total pages: 15
  - Text blocks: 342
  - Content regions: 8

[STEP 2] Processing text into chunks...
✓ Created 47 chunks
  - First chunk: sample_chunk_0
  - Average tokens per chunk: 478.2
  - Chunks with statistics: 5

[STEP 3] Connecting to Weaviate...
✓ Connected to Weaviate

[STEP 4] Creating schema...
✓ Schema created

[STEP 5] Inserting chunks into vector store...
✓ Chunks inserted

[STEP 6] Verifying insertion...
✓ Collection stats:
  - TextChunk: 47 entries
  - Table: 0 entries
  - Figure: 0 entries

[STEP 7] Testing search...

Query: 'What is the main topic of this document?'
  Result 1:
    - Chunk: sample_chunk_0
    - Score: 0.8542
    - Page: 0
    - Section: Introduction
    - Preview: This document presents...

============================================================
✓ PHASE 1 PIPELINE TEST COMPLETE
============================================================
```

## Next Steps After Successful Test

Once Phase 1 test passes:
- ✓ Basic pipeline working
- Ready to proceed to Phase 2: Advanced Extraction (tables & figures)
