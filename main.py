from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse, HTMLResponse
import os
import asyncio
import base64
import json
from pathlib import Path
import uuid

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from utils import astream_graph
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from services.streaming_service import (
    RealTimeAgentStreamer
)
from dotenv import load_dotenv

from services.mem0_service import Mem0ConversationService
from services.streaming_service import Mem0StreamingService

load_dotenv()

real_time_streamer = RealTimeAgentStreamer()
TEMPLATES_DIR = Path("./templates")
TEMPLATES_DIR.mkdir(exist_ok=True)
# ----------------------------
# FastAPI Setup
# ----------------------------
app = FastAPI(title="LangChain MCP RAG API", version="1.0.0")

# ----------------------------
# File Storage Configuration
# ----------------------------
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ----------------------------
# Environment Setup
# ----------------------------
# Load Azure OpenAI configuration for chat/completion only
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_MODEL_API_VERSION = os.getenv("AZURE_OPENAI_MODEL_API_VERSION")

DOCUMENT_MCP_URL = "http://localhost:8001/sse"
DOCDB_SUMMARIZATION_MCP_URL = "http://localhost:8003/sse"
RAG_MCP_URL = "http://localhost:8002/sse"

# ----------------------------
# Global Variables
# ----------------------------
agent = None
mcp_client = None
memory_service = None
streaming_service = None


# ----------------------------
# Model and Agent Setup (once on startup)
# ----------------------------
@app.on_event("startup")
async def setup_agent():
    global agent, mcp_client, memory_service, streaming_service
    print("Setting up agent and MCP client")
    model = AzureChatOpenAI(
        model_name=AZURE_OPENAI_MODEL_NAME,
        openai_api_version=AZURE_OPENAI_MODEL_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        temperature=0.1
    )

    # Connect to Document MCP server
    mcp_client = MultiServerMCPClient({

        "DocDBSummarizationService": {
            "url": DOCDB_SUMMARIZATION_MCP_URL,
            "transport": "sse",
        },
        "RAGService": {
            "url": RAG_MCP_URL,
            "transport": "sse",
        }
    })

    memory_service = Mem0ConversationService()
    streaming_service = Mem0StreamingService(memory_service)

    tools = await mcp_client.get_tools()
    AGENT_PROMPT = """
    You are an AI assistant with connection to these services:
    - DocDBSummarizationService: for summarizing documents, given a document_id
    - RAGService: for querying the vector database

    If the user ask you to summary a document, if they provide a document id, you only use the DocDBSummarizationService to summarize documents.
    If the user ask you to answer a question, you only use the RAGService to query the vector database and answer questions related to user's personal documents.
    """
    agent = create_react_agent(model, tools, prompt=AGENT_PROMPT)


@app.post("/agent")
async def agent_endpoint(query: str = Form(...)):
    """Query the vector database"""
    answer = await astream_graph(
        agent, {"messages": query}
    )
    return answer


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    html_file_path = TEMPLATES_DIR / "index.html"
    try:
        with open(html_file_path, "r", encoding='utf-8') as file:
            html = file.read()
            return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/{session_id}/stream")
async def stream_chat_with_session(session_id: str, message: str = Form(...)):
    if not streaming_service:
        raise HTTPException(status_code=500, detail="No streaming service")
    return StreamingResponse(
        streaming_service.generate_memory_stream(agent, message=message, session_id=session_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


@app.post("/agent/stream/realtime")
async def stream_agent_thinking_realtime(query: str = Form(...)):
    """
    Stream the agent's thinking process (real-time version)

    Args:
        query: The user's question/query sent as form data

    Returns:
        StreamingResponse with real-time agent thinking data
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    print(f"[DEBUG] Received query: {query}")

    return StreamingResponse(
        real_time_streamer.generate_real_time_stream(agent, query),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


# ----------------------------
# Document Management Routes
# ----------------------------
@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file"""
    try:
        # Generate unique document ID
        document_id = str(uuid.uuid4())

        # Save file to local storage
        file_content = await file.read()
        file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

        with open(file_path, 'wb') as f:
            f.write(file_content)

        # Call document service via MCP with file path
        async with sse_client(DOCUMENT_MCP_URL) as (read_stream, write_stream):
            # Create a client session
             async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()
                print("Connected to DocumentService MCP server")

                # Call the process_document tool
                result = await session.call_tool(
                    "process_document",
                    arguments={
                        "file_path": file_path,
                        "filename": file.filename,
                        "document_id": document_id
                    }
                )

        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "processing_result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload-mongo")
async def upload_document_mongo(file: UploadFile = File(...)):
    """Upload a document and save text to MongoDB only (no vectorization)"""
    try:
        document_id = str(uuid.uuid4())
        file_content = await file.read()
        file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(file_content)
        # Call the new tool via MCP
        async with sse_client(DOCUMENT_MCP_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "upload_and_save_to_mongo",
                    arguments={
                        "file_path": file_path,
                        "filename": file.filename,
                        "document_id": document_id
                    }
                )
        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "processing_result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
#  RAG Routes
# ----------------------------

# ----------------------------
# Health Check
# ----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "mcp_client_initialized": mcp_client is not None
    }
