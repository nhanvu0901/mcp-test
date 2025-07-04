import os
import sys
from typing import Dict, Any
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings
from services.document_processor import DocumentProcessor
from pymongo import MongoClient
project_root = Path(__file__).parent.parent
os.chdir(project_root)
# Initialize MCP server
mcp = FastMCP(
    "DocumentService",
    instructions="Document processing service that can upload, process, and manage documents with vector embeddings.",
    host="0.0.0.0",
    port=8001,
)

# Global configurations
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
COLLECTION_NAME = "RAG"
VECTOR_SIZE = 3072

# Initialize MongoDB client
mongo_uri = os.getenv("MONGODB_URI")
mongo_client = MongoClient(mongo_uri) if mongo_uri else None

# Initialize external services
qdrant_client = QdrantClient(host="localhost", port=6333)

# Load Azure OpenAI configuration from environment variables
azure_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
azure_embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
azure_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
azure_embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION")

# Create embedding model
embedding_model = AzureOpenAIEmbeddings(
    model=azure_embedding_model,
    azure_endpoint=azure_embedding_endpoint,
    api_key=azure_embedding_api_key,
    openai_api_version=azure_embedding_api_version
)

           
# Initialize DocumentProcessor with external configuration
document_processor = DocumentProcessor(
    collection_name=COLLECTION_NAME,
    qdrant_host="localhost",
    qdrant_port=6333,
    embedding_model=embedding_model,
    vector_size=VECTOR_SIZE,
    mongo_client=mongo_client
)

# THIS WOULD BE CALLED DIRECTLY FROM THE APP
@mcp.tool()
async def process_document(
    file_path: str,
    filename: str,
    document_id: str,
) -> Dict[str, Any]:
    """
    Process a document from file path, extract text, create embeddings, and store in vector database.
    
    Args:
        file_path (str): Path to the document file
        filename (str): Original filename with extension
        document_id (str): Unique document ID
            
    Returns:
        dict: Processing result with document_id and status
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
                "document_id": document_id
            }
        
        # Extract file type
        file_type = filename.split('.')[-1].lower()
        
        # Extract text using DocumentProcessor
        text_content = document_processor.extract_text(file_path)
        
        # Use DocumentProcessor to process and add chunks to Qdrant
        document_processor.process_and_add_chunks_to_qdrant(
            text=text_content,
            method="auto",
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
            file_type=file_type,
            document_name=filename,
            document_id=document_id
        )
        
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "file_path": file_path,
            "collection_id": COLLECTION_NAME
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "document_id": document_id
        }

@mcp.tool()
async def upload_and_save_to_mongo(
    file_path: str,
    filename: str,
    document_id: str,
) -> Dict[str, Any]:
    """
    Upload a document, extract text, and save to MongoDB only (no vectorization).
    """
    try:
        # Validate file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
                "document_id": document_id
            }
        file_type = filename.split('.')[-1].lower()
        # Extract and save to MongoDB
        text = document_processor.extract_and_save_to_mongo(
            file_path=file_path,
            document_id=document_id,
            document_name=filename,
            file_type=file_type
        )
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "file_path": file_path,
            "mongo_saved": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "document_id": document_id
        }

if __name__ == "__main__":
    print("Document Service MCP server is running on port 8001...")
    mcp.run(transport="sse")