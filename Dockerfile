# RAG Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libxcb + libgl for Docling/PyMuPDF PDF processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy protobuf files and generate Python code using grpcio-tools
# (guarantees generated code matches the installed protobuf runtime)
COPY protos/ protos/
RUN mkdir -p gen/python/rag/v1 && \
    touch gen/__init__.py gen/python/__init__.py gen/python/rag/__init__.py gen/python/rag/v1/__init__.py && \
    python -m grpc_tools.protoc \
        --proto_path=protos \
        --python_out=gen/python \
        --grpc_python_out=gen/python \
        rag/v1/rag_service.proto

# Copy source code
COPY src/ src/

# Set Python path
ENV PYTHONPATH=/app/src:/app:/app/gen/python

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=5)"

# Run the server
CMD ["python", "-m", "rag_service.main"]
