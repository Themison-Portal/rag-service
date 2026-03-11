# RAG Service

gRPC microservice for RAG (Retrieval-Augmented Generation) in the Themison clinical trials platform.

## Architecture

```
Frontend (React) ──REST──> Backend (FastAPI) ──gRPC──> RAG Microservice
                                 │
                                 ├── Redis (caching layer)
                                 └── PostgreSQL (shared database)
```

## Features

- **PDF Ingestion**: Parse PDFs using Docling, chunk with HybridChunker, embed with OpenAI
- **Hybrid Search**: Vector similarity (pgvector) + BM25 full-text search with RRF fusion
- **Semantic Cache**: pgvector similarity-based query caching
- **LLM Generation**: Claude Sonnet for answer generation with citations
- **PDF Highlighting**: Generate highlighted PDF pages with bounding boxes

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 16 with pgvector extension

### Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate protobuf code**:
   ```bash
   python scripts/generate_proto.py
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start services**:
   ```bash
   docker-compose up -d
   ```

5. **Run the gRPC server**:
   ```bash
   python -m rag_service.main
   ```

### Docker Deployment

```bash
docker-compose up -d --build
```

## gRPC API

### IngestPdf

Ingest a PDF document with streaming progress updates.

```protobuf
rpc IngestPdf(IngestPdfRequest) returns (stream IngestPdfProgress);
```

### Query

Execute RAG query - retrieve chunks and generate answer.

```protobuf
rpc Query(QueryRequest) returns (QueryResponse);
```

### GetHighlightedPdf

Get a PDF page with highlighted bounding boxes.

```protobuf
rpc GetHighlightedPdf(GetHighlightedPdfRequest) returns (HighlightedPdfResponse);
```

### InvalidateDocument

Invalidate cache entries for a document (called on re-upload).

```protobuf
rpc InvalidateDocument(InvalidateDocumentRequest) returns (InvalidateDocumentResponse);
```

### HealthCheck

Check service health status.

```protobuf
rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
```

## Testing with grpcurl

```bash
# Health check
grpcurl -plaintext localhost:50051 themison.rag.v1.RagService/HealthCheck

# List services
grpcurl -plaintext localhost:50051 list
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude | Required |
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `GRPC_PORT` | gRPC server port | 50051 |
| `GRPC_MAX_WORKERS` | Max gRPC worker threads | 10 |
| `HYBRID_SEARCH_ENABLED` | Enable hybrid search | true |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | Cache similarity threshold | 0.90 |
| `LLM_MODEL` | Claude model to use | claude-sonnet-4-20250514 |

## Project Structure

```
rag-service/
├── protos/
│   └── rag/v1/rag_service.proto      # gRPC service definition
├── gen/python/                        # Generated protobuf code
├── src/rag_service/
│   ├── main.py                        # gRPC server entry point
│   ├── server.py                      # RagServicer implementation
│   ├── config.py                      # Pydantic settings
│   ├── services/
│   │   ├── ingestion_service.py       # PDF ingestion pipeline
│   │   ├── retrieval_service.py       # Vector/hybrid search
│   │   ├── generation_service.py      # LLM generation
│   │   └── highlighting_service.py    # PDF highlighting
│   ├── cache/
│   │   └── semantic_cache.py          # pgvector semantic cache
│   ├── models/
│   │   ├── chunks.py                  # DocumentChunkDocling
│   │   └── semantic_cache.py          # SemanticCacheResponse
│   └── db/session.py                  # Async SQLAlchemy
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Integration with Backend

Set these environment variables in the FastAPI backend to use the gRPC RAG service:

```bash
RAG_SERVICE_ADDRESS=localhost:50051
USE_GRPC_RAG=true
```

The backend will automatically route requests to the gRPC service when enabled.
