-- Initialize PostgreSQL with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create document_chunks_docling table if not exists
CREATE TABLE IF NOT EXISTS document_chunks_docling (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_metadata JSONB,
    embedding vector(1536),
    embedding_large vector(2000),
    contextual_summary TEXT,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for document_chunks_docling
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON document_chunks_docling
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks_docling (document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_content_gin ON document_chunks_docling USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_large_hnsw ON document_chunks_docling
    USING hnsw (embedding_large vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Create semantic_cache_responses table if not exists
CREATE TABLE IF NOT EXISTS semantic_cache_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    query_embedding vector(1536) NOT NULL,
    document_id UUID NOT NULL,
    response_data JSONB NOT NULL,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    context_hash VARCHAR(32) NOT NULL
);

-- Create indexes for semantic_cache_responses
CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding_hnsw ON semantic_cache_responses
    USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_document_id ON semantic_cache_responses (document_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
