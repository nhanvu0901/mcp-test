import pymupdf4llm
from docx import Document
import pandas as pd
import os
import litellm
from pymongo import MongoClient



class LLMClient:
    """LLM Client for OpenAI/Azure integration"""
    
    def __init__(self, model_name: str, api_key: str, num_retries: int = 3, **kwargs):
        # Prepares the arguments for API calls
        self.default_args = {
            "model": model_name,
            "api_key": api_key,
            "num_retries": num_retries,
            **kwargs,
        }

        # Add Azure-specific arguments if needed
        if model_name.startswith("azure/"):
            api_base = kwargs.get("api_base", None)
            api_version = kwargs.get("api_version", None)
            if api_base and api_version:
                self.default_args["api_base"] = api_base
                self.default_args["api_version"] = api_version
            else:
                raise ValueError(
                    "Both `api_base` and `api_version` must be provided for Azure models."
                )

    def complete(self, messages, **kwargs):
        args = {**self.default_args, "messages": messages, **kwargs}
        return litellm.completion(**args)

    async def acomplete(self, messages, **kwargs):
        args = {**self.default_args, "messages": messages, **kwargs}
        return await litellm.acompletion(**args)

    def stream(self, messages, **kwargs):
        args = {**self.default_args, "messages": messages, **kwargs}
        return litellm.completion(**args, stream=True)

    async def astream(self, messages, **kwargs):
        args = {**self.default_args, "messages": messages, **kwargs}
        return await litellm.acompletion(**args, stream=True)


def get_llm_client() -> LLMClient:
    """Get LLM client based on environment configuration"""
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if azure_api_key:
        return LLMClient(
            f"azure/{os.getenv('AZURE_OPENAI_MODEL_NAME')}",
            api_key=azure_api_key,
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_MODEL_API_VERSION"),
        )
    elif openai_api_key:
        return LLMClient(
            f"openai/{os.getenv('OPENAI_MODEL_NAME', 'gpt-4o-mini')}",
            api_key=openai_api_key,
        )
    else:
        raise EnvironmentError(
            "No API key found. Set either AZURE_OPENAI_API_KEY or OPENAI_API_KEY in environment."
        )


def extract_text_from_pdf(file_path: str, 
                          pages: list[int] = None,
                          write_images: bool = False,
                          image_path: str = None) -> str:
    """Extract text from PDF file as MD format"""
    md_text = pymupdf4llm.to_markdown(file_path, pages=pages, write_images=write_images, image_path=image_path)
    return md_text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    md_lines = []

    for para in doc.paragraphs:
        style = para.style.name
        text = ""

        if style.startswith("Heading"):
            level = int(style.split()[-1]) 
            text = f"{'#' * level} {para.text}"
        elif style in ["List Paragraph"]:
            text = f"- {para.text}"
        else:
            for run in para.runs:
                run_text = run.text
                if not run_text.strip():
                    text += run_text
                    continue
                if run.bold:
                    run_text = f"**{run_text}**"
                if run.italic:
                    run_text = f"*{run_text}*"
                if run.underline:
                    run_text = f"_{run_text}_"
                text += run_text

        md_lines.append(text)
    md_text = "\n\n".join(md_lines)
    return md_text

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def extract_text_from_csv(file_path: str) -> str:
    """Extract text from CSV file"""
    df = pd.read_csv(file_path)
    return df.to_string()

def extract_text_from_md(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_py(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_tex(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text_from_html(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path: str) -> str:
    """Extract text based on file type"""
    file_type = file_path.split('.')[-1]
    if file_type.lower() == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_type.lower() in ['docx', 'doc']:
        return extract_text_from_docx(file_path)
    elif file_type.lower() == 'txt':
        return extract_text_from_txt(file_path)
    elif file_type.lower() == 'csv':
        return extract_text_from_csv(file_path)
    elif file_type.lower() == 'md':
        return extract_text_from_md(file_path)
    elif file_type.lower() == 'py':
        return extract_text_from_py(file_path)
    elif file_type.lower() == 'tex':
        return extract_text_from_tex(file_path)
    elif file_type.lower() == 'html':
        return extract_text_from_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def get_document_text(mongo_client: MongoClient, document_id: str) -> str:
    """
    Retrieve the text field from a MongoDB document by document_id.
    Requires MONGODB_DB, MONGODB_COLLECTION in environment.
    """
    if not mongo_client:
        raise ValueError("MongoDB client is required")
    db_name = os.getenv("MONGODB_DB")
    collection_name = os.getenv("MONGODB_COLLECTION")
    if not (db_name and collection_name):
        raise ValueError("Missing MongoDB database/collection info in environment variables.")
    db = mongo_client[db_name]
    collection = db[collection_name]
    doc = collection.find_one({"_id": document_id})
    if not doc or "text" not in doc:
        raise ValueError("Document not found or missing 'text' field")
    return doc["text"]
