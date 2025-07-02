 #!/bin/bash
set -e


cleanup() {
    echo "Cleaning up..."
    if [ ! -z "$DOCUMENT_PID" ] && kill -0 $DOCUMENT_PID 2>/dev/null; then
        echo "Stopping Document Server (PID: $DOCUMENT_PID)"
        kill $DOCUMENT_PID
    fi
    if [ ! -z "$RAG_PID" ] && kill -0 $RAG_PID 2>/dev/null; then
        echo "Stopping RAG Server (PID: $RAG_PID)"
        kill $RAG_PID
    fi
    if [ ! -z "$DOCDB_PID" ] && kill -0 $DOCDB_PID 2>/dev/null; then
        echo "Stopping DocDB Summarization Server (PID: $DOCDB_PID)"
        kill $DOCDB_PID
    fi
    exit 0
}


trap cleanup SIGTERM SIGINT

echo "=== Starting AI Assistant Application ==="


echo "Starting MCP Document Server..."
python mcp_servers/mcp_server_document.py &
DOCUMENT_PID=$!
echo "Document Server started with PID: $DOCUMENT_PID"

echo "Starting MCP RAG Server..."
python mcp_servers/mcp_server_rag.py &
RAG_PID=$!
echo "RAG Server started with PID: $RAG_PID"

echo "Starting MCP DocDB Summarization Server..."
python mcp_servers/mcp_server_docdb_summarization.py &
DOCDB_PID=$!
echo "DocDB Summarization Server started with PID: $DOCDB_PID"


echo "Waiting for MCP servers to initialize..."
sleep 10

echo "Checking MCP server health..."
for i in {1..30}; do
    if curl -f http://localhost:8001/health 2>/dev/null && \
       curl -f http://localhost:8002/health 2>/dev/null && \
       curl -f http://localhost:8003/health 2>/dev/null; then
        echo "All MCP servers are healthy!"
        break
    fi
    echo "Waiting for MCP servers... (attempt $i/30)"
    sleep 2
done

# Start FastAPI application
echo "Starting FastAPI application..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload