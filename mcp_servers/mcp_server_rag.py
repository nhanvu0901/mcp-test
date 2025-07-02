from typing import  List

import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

# Initialize MCP server
mcp = FastMCP(
    "RAGService",
    instructions="This is a RAG (Retrieval-Augmented Generation) service that can search and retrieve relevant document chunks based on queries.",
    host="0.0.0.0",
    port=8002,
)

# Global configurations
COLLECTION_NAME = "RAG"
# Initialize external services
qdrant_client = QdrantClient(host="localhost", port=6333)


# Load Azure OpenAI configuration from environment variables
azure_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
azure_embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
azure_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
azure_embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION")

embedding_model = AzureOpenAIEmbeddings(
    model=azure_embedding_model,
    azure_endpoint=azure_embedding_endpoint,
    api_key=azure_embedding_api_key,
    openai_api_version=azure_embedding_api_version
)


@mcp.tool()
async def retrieve(query: str, limit: int = 5) -> str:
    """
    Query the Qdrant vector database with a text query and return matching results.
    
    Args:
        query_text (str): The text query to search for
        limit (int): Maximum number of results to return (default: 5)
    
    Returns:
        str: Concatenated text content from all retrieved documents
    """
    try:
        # Generate embedding for the query text
        query_embedding = embedding_model.embed_query(query)
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit
        )
        
        # Extract text content from results
        results = []
        for result in search_results:
            # Assuming the text content is stored in payload under 'text' key
            # Adjust the key name based on your actual data structure
            if 'text' in result.payload:
                results.append(result.payload['text'])
            elif 'content' in result.payload:
                results.append(result.payload['content'])
            else:
                # If no text field found, convert payload to string
                results.append(str(result.payload))
        
        return "\n".join(results)
        
    except Exception as e:
        print(f"Error during query: {e}")
        return []

if __name__ == "__main__":
    print("RAG Service MCP server is running on port 8002...")
    mcp.run(transport="sse")