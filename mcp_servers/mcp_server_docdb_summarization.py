"""
MCP Server for DocumentDB-based Summarization/Translation

This server provides summarization and translation capabilities by retrieving text from MongoDB using document_id.
"""

import os
import sys
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pymongo import MongoClient

# Add parent directory to path to import services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import (
    summarize_text_with_detail_level,
    summarize_text_with_word_count
)
from services.utils import get_document_text, get_llm_client

# Load environment variables
load_dotenv()

# Initialize MongoDB client
mongo_uri = os.getenv("MONGODB_URI")
mongo_client = MongoClient(mongo_uri) if mongo_uri else None

# Initialize MCP server
mcp = FastMCP(
    "DocDBSummarizerService",
    instructions="Summarize or translate text by document_id, retrieving text from MongoDB.",
    host="0.0.0.0",
    port=8003,
)

llm = get_llm_client()

@mcp.tool()
async def summarize_by_detail_level(
    document_id: str,
    summarization_level: str = "medium",
    further_instruction: str = None
) -> str:
    """
    Summarize text from MongoDB by document_id with detail level.

    Args:
        document_id: The ID of the document to summarize.
        summarization_level: The level of detail for the summary, can be "concise", "medium" or "detailed".
        further_instruction: Further instructions for the summary, can be "translate to target language", "translate to target language and summarize", "translate to target language and summarize with detail level", "translate to target language and summarize with detail level and further instructions".

    Returns:
        A summary of the document.
    """
    try:
        text = get_document_text(mongo_client, document_id)
        summary, word_count = await summarize_text_with_detail_level(
            text=text,
            summarization_level=summarization_level,
            further_instruction=further_instruction
        )
        return f"Summary ({word_count} words):\n\n{summary}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def summarize_by_word_count(
    document_id: str,
    num_words: int = 100
) -> str:
    """
    Summarize text from MongoDB by document_id to a target word count.

    Args:
        document_id: The ID of the document to summarize.
        num_words: The target word count for the summary.

    Returns:
        A summary of the document.
    """
    try:
        text = get_document_text(mongo_client, document_id)
        summary, actual_word_count = await summarize_text_with_word_count(
            text=text,
            num_words=num_words
        )
        return f"Summary (Target: {num_words} words, Actual: {actual_word_count} words):\n\n{summary}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("DocDB Summarizer MCP server is running on port 8004...")
    mcp.run(transport="sse") 