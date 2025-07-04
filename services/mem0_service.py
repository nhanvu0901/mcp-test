import os
from typing import List, Any, Coroutine
from mem0 import Memory
from mem0.configs.base import MemoryConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.embeddings.configs import EmbedderConfig
from dotenv import load_dotenv

load_dotenv()


def get_mem0_config() -> MemoryConfig:
    """Configure Mem0AI to use your existing infrastructure"""
    llm_config = LlmConfig(
        provider="azure_openai",
        config={
            "model": os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini"),
            "temperature": 0.1,
            "azure_kwargs": {
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "api_version": os.getenv("AZURE_OPENAI_MODEL_API_VERSION"),
                "azure_deployment": os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
            }
        }
    )

    vector_store_config = VectorStoreConfig(
        provider="qdrant",
        config={
            "collection_name": "mem0_memories",
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
            "embedding_model_dims": 3072
        }
    )

    embedder_config = EmbedderConfig(
        provider="azure_openai",
        config={
            "model": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            "azure_kwargs": {
                "azure_endpoint": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
                "api_version": os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION"),
                "azure_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            }
        }
    )

    memory_config = MemoryConfig(
        vector_store=vector_store_config,
        llm=llm_config,
        embedder=embedder_config,
        version="v1.1"
    )

    return memory_config


class Mem0ConversationService:

    def __init__(self):
        self.memory = Memory(config=get_mem0_config())

    async def chat_with_memory(self, message: str, session_id: str = "default") -> tuple[str, list[dict[str, str]]]:
        memories = self.memory.search(query=message, user_id=session_id, limit=5)
        memories_text = "\n".join([f"- {mem['memory']}" for mem in memories.get("results", [])])

        system_context = ""
        if memories_text:
            system_context = f"\n\nRelevant conversation history and user context:\n{memories_text}"

        conversation = [
            {"role": "system", "content": f"You are a helpful RAG assistant.{system_context}"},
            {"role": "user", "content": message}
        ]

        return memories_text, conversation

    async def save_conversation_memory(self, user_message: str, agent_response: str, session_id: str):
        conversation = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": agent_response}
        ]
        self.memory.add(conversation, user_id=session_id)

    async def get_session_memories(self, session_id: str) -> List[dict]:
        try:
            memories = self.memory.get_all(user_id=session_id)
            return memories.get("results", [])
        except Exception as e:
            print(f"Error getting memories: {e}")
            return []

    async def delete_session_memories(self, session_id: str) -> bool:
        try:
            memories = await self.get_session_memories(session_id)
            for memory in memories:
                if 'id' in memory:
                    self.memory.delete(memory['id'])
            return True
        except Exception as e:
            print(f"Error deleting memories: {e}")
            return False