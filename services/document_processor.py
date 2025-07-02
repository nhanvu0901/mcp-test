from langchain_openai import AzureOpenAIEmbeddings
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from pymongo.errors import OperationFailure

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    PythonCodeTextSplitter,
    LatexTextSplitter
)

from services.utils import extract_text

import os
from pymongo import MongoClient

# Default configurations
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_COLLECTION_NAME = "mcp"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
VECTOR_SIZE = 3072

# MongoDB config from environment
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")

def save_document_to_mongo(mongo_client: MongoClient, document_id: str, text: str, metadata: dict = None):
    """Save or update a document in MongoDB with _id=document_id and text."""
    if not mongo_client:
        print("[MongoDB] No client provided, skipping save.")
        return
    if not (MONGODB_DB and MONGODB_COLLECTION):
        print("[MongoDB] Missing database/collection info, skipping save.")
        return
    try:
        db = mongo_client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]
        doc = {
            "_id": document_id,
            "text": text,
        }
        if metadata:
            doc.update(metadata)
        result = collection.replace_one({"_id": document_id}, doc, upsert=True)
        if result.upserted_id or result.modified_count > 0:
            print(f"[MongoDB] Saved document_id={document_id} to MongoDB")
            return True
        else:
            print(f"[MongoDB] No changes made for document_id={document_id}")
            return True
    except OperationFailure as e:
        print(f"[MongoDB] Operation failed for document_id={document_id}: {e}")
        return False


class ChunkingMethod:
    """Available chunking methods with their configurations"""
    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    TOKEN = "token"
    SPACY = "spacy"
    NLTK = "nltk"
    MARKDOWN_HEADER = "markdown_header"
    HTML_HEADER = "html_header"
    PYTHON_CODE = "python_code"
    LATEX = "latex"
    # Added for consistency, although not fully implemented in your provided code
    CUSTOM_TOKEN = "custom_token" 


class DocumentProcessor:
    def __init__(self, 
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 qdrant_host: str = DEFAULT_QDRANT_HOST,
                 qdrant_port: int = DEFAULT_QDRANT_PORT,
                 embedding_model: AzureOpenAIEmbeddings = None,
                 vector_size: int = VECTOR_SIZE,
                 mongo_client: MongoClient = None):
        """
        Initialize DocumentProcessor with configuration parameters.
        
        Args:
            collection_name: Name of the Qdrant collection
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            embedding_model: Pre-configured AzureOpenAIEmbeddings instance
            vector_size: Expected vector dimension for the collection
            mongo_client: MongoDB client instance
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.mongo_client = mongo_client
        
        if not self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' does not exist. Creating it.")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Collection '{self.collection_name}' already exists.")

        # Use provided embedding model or raise error if not provided
        if embedding_model is None:
            raise ValueError("embedding_model must be provided. Please configure AzureOpenAIEmbeddings externally and pass it to DocumentProcessor.")
        
        self.embedding_model = embedding_model
    
    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract text from file"""
        return extract_text(file_path)

    @staticmethod
    def get_text_splitter(
        method: str = ChunkingMethod.RECURSIVE_CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        **kwargs
    ):
        """Get the appropriate text splitter based on method"""
        
        if method == ChunkingMethod.RECURSIVE_CHARACTER:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
                **kwargs
            )
        
        elif method == ChunkingMethod.CHARACTER:
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=kwargs.get('separator', '\n\n'),
                **kwargs
            )
        
        elif method == ChunkingMethod.TOKEN:
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name=kwargs.get('encoding_name', 'cl100k_base'),
                **kwargs
            )
        
        elif method == ChunkingMethod.SPACY:
            return SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
        elif method == ChunkingMethod.NLTK:
            return NLTKTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
        elif method == ChunkingMethod.MARKDOWN_HEADER:
            headers_to_split_on = kwargs.get('headers_to_split_on', [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ])
            return MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                **kwargs
            )
        
        elif method == ChunkingMethod.HTML_HEADER:
            headers_to_split_on = kwargs.get('headers_to_split_on', [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ])
            return HTMLHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on,
                **kwargs
            )
        
        elif method == ChunkingMethod.PYTHON_CODE:
            return PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
        elif method == ChunkingMethod.LATEX:
            return LatexTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
        else:
            # Default to recursive character splitter
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs
            )
        
    
    # Changed to instance method as it relies on self.client and self.embedding_model
    def process_and_add_chunks_to_qdrant(self, 
        text: str, 
        method: str = ChunkingMethod.RECURSIVE_CHARACTER,
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        file_type: str = None,
        document_name: str = None,
        document_id: str = None,  # Added document_id parameter
        **kwargs
    ) -> None:
        """Split text into chunks using various methods and add them to Qdrant"""
        

        # Auto-select method based on file type if not specified
        if method == "auto":
            if file_type == "md":
                method = ChunkingMethod.MARKDOWN_HEADER
            elif file_type == "py":
                method = ChunkingMethod.PYTHON_CODE
            elif file_type == "tex":
                method = ChunkingMethod.LATEX
            elif file_type == "html":
                method = ChunkingMethod.HTML_HEADER
                
            else:
                method = ChunkingMethod.RECURSIVE_CHARACTER
            
        # Get the appropriate splitter
        splitter = self.get_text_splitter(method, chunk_size, overlap, **kwargs)
        
        # Handle special cases for header-based splitters
        if method in [ChunkingMethod.MARKDOWN_HEADER, ChunkingMethod.HTML_HEADER]:
            chunks = splitter.split_text(text)
            # Convert Document objects to strings if needed
            chunks = [chunk.page_content if hasattr(chunk, 'page_content') else str(chunk) for chunk in chunks]
        else:
            # Standard text splitting
            chunks = splitter.split_text(text)
        
        # Add each chunk to Qdrant
        points_to_upsert = []
        for i, chunk in enumerate(chunks):
            # Embed documents accepts a list, and returns a list of embeddings.
            # We want the first (and only) embedding for the current chunk.
            try:
                embedded_text = self.embedding_model.embed_documents([chunk])[0] 
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                continue # Skip this chunk if embedding fails

            metadata = {
                "chunk_id": i,
                "document_name": document_name,
                "text": chunk, 
                "chunk_method": method,
                "file_type": file_type,
                "document_id": document_id 
            }
            points_to_upsert.append(PointStruct(id=str(uuid.uuid4()), vector=embedded_text, payload=metadata))
        
        if points_to_upsert:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    wait=True, # Wait for operation to complete
                    points=points_to_upsert
                )
                print(f"Successfully upserted {len(points_to_upsert)} chunks to Qdrant for document ID: {document_id}")
            except Exception as e:
                print(f"Error upserting points to Qdrant for document ID {document_id}: {e}")
        else:
            print(f"No chunks to upsert for document ID: {document_id}")

        # Save to MongoDB after extracting text (if document_id is provided)
        if document_id and self.mongo_client:
            save_document_to_mongo(self.mongo_client, document_id, text, metadata={
                "document_name": document_name,
                "file_type": file_type
            })

    def extract_and_save_to_mongo(
        self,
        file_path: str,
        document_id: str,
        document_name: str = None,
        file_type: str = None,
        metadata: dict = None
    ):
        """Extract text from file and save to MongoDB only (no vectorization)."""
        text = extract_text(file_path)
        meta = metadata or {}
        meta.update({
            "document_name": document_name,
            "file_type": file_type
        })
        save_document_to_mongo(self.mongo_client, document_id, text, meta)
        print(f"[MongoDB] Only saved extracted text for document_id={document_id}")
        return text


# if __name__ == "__main__":
#     # Example usage - configuration should be loaded externally
#     # Load Azure OpenAI configuration from environment variables
#     azure_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
#     azure_embedding_api_key = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
#     azure_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
#     azure_embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

#     # Create embedding model
#     embedding_model = AzureOpenAIEmbeddings(
#         model=azure_embedding_model,
#         azure_endpoint=azure_embedding_endpoint,
#         api_key=azure_embedding_api_key,
#         openai_api_version=azure_embedding_api_version
#     )

#     # Create a dummy mcp.md file for demonstration if it doesn't exist
#     document_path = "mcp.md"
#     if not os.path.exists(document_path):
#         print(f"Creating a dummy file: {document_path}")
#         with open(document_path, "w", encoding="utf-8") as f:
#             f.write("# Meeting Minutes\n\n")
#             f.write("## June 19, 2025\n\n")
#             f.write("### Attendees\n")
#             f.write("- Alice\n")
#             f.write("- Bob\n")
#             f.write("- Charlie\n\n")
#             f.write("### Discussion Points\n")
#             f.write("1.  **Project Alpha**: Reviewed progress. On track for phase 1 completion.\n")
#             f.write("2.  **Budget Review**: Discussed Q2 expenditures. Need to optimize cloud spending.\n")
#             f.write("3.  **New Initiatives**: Brainstormed ideas for next quarter. Focus on AI integration.\n\n")
#             f.write("### Action Items\n")
#             f.write(" - Alice: Prepare a detailed report on cloud spending by EOD.\n")
#             f.write(" - Bob: Research potential AI integration partners.\n")
#             f.write(" - Charlie: Schedule a follow-up meeting for next week.\n")
#     else:
#         print(f"Using existing file: {document_path}")

#     processor = DocumentProcessor(
#         collection_name="mcp",
#         qdrant_host="localhost",
#         qdrant_port=6333,
#         embedding_model=embedding_model,
#         vector_size=VECTOR_SIZE  # Adjust based on your embedding model
#     )

#     document_name = document_path

#     try:
#         # Extract text from the markdown file
#         print(f"Extracting text from {document_path}...")
#         document_text = processor.extract_text(document_path)
#         print(f"Text extracted (first 200 chars): {document_text[:200]}...")
#         # Process and add chunks to Qdrant
#         print(f"Processing and adding chunks to Qdrant using '{ChunkingMethod.MARKDOWN_HEADER}' method...")
#         # Use 'auto' or 'markdown_header' explicitly for .md files
#         processor.process_and_add_chunks_to_qdrant(
#             text=document_text,
#             method="auto", # This will correctly identify markdown
#             file_type="md",
#             document_name=document_name,
#         )
#         print("Document processing complete.")

#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred during processing: {e}")