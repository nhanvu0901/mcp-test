from typing import Dict, Any, AsyncGenerator
import asyncio
import json
from datetime import datetime

from utils import astream_graph
from .mem0_service import Mem0ConversationService


class RealTimeAgentStreamer:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.is_streaming = False

    async def real_time_callback(self, data: Dict[str, Any]):
        node = data.get("node", "Agent")
        content = data.get("content", "")
        metadata = data.get("metadata", {})

        content_text = ""
        message_type = "thinking"

        try:
            if hasattr(content, 'content'):
                if isinstance(content.content, list):
                    for item in content.content:
                        if isinstance(item, dict) and "text" in item:
                            content_text += item["text"]
                elif isinstance(content.content, str):
                    content_text = content.content
                else:
                    content_text = str(content.content)

                if hasattr(content, 'tool_calls') and content.tool_calls:
                    message_type = "tool_call"
                    tool_names = []
                    for tool in content.tool_calls:
                        if hasattr(tool, 'name') and tool.name:
                            tool_names.append(tool.name)
                        elif isinstance(tool, dict) and tool.get('name'):
                            tool_names.append(tool.get('name'))

                    if tool_names:
                        content_text = f"Calling tools: {', '.join(tool_names)}"
                    else:
                        content_text = "Calling tool..."

                elif not content_text and node.lower() == 'agent':
                    step = metadata.get('langgraph_step', 0)
                    if step <= 2:
                        message_type = "reasoning"
                        reasoning_messages = [
                            "Analyzing your question...",
                            "Thinking ....",
                            "Determining which tools to use...",
                            "Processing your request...",
                            "Planning the response strategy..."
                        ]
                        content_text = reasoning_messages[step % len(reasoning_messages)]

            if not content_text:
                if isinstance(content, str) and content.strip():
                    content_text = content
                elif not content_text:
                    return

            chunk = {
                "type": message_type,
                "node": node,
                "content": content_text,
                "step": metadata.get('langgraph_step', 0),
                "timestamp": asyncio.get_event_loop().time()
            }
            await self.queue.put(chunk)

        except Exception as e:
            print(f"Error in real_time_callback: {e}")
            error_chunk = {
                "type": "error",
                "node": node,
                "content": f"Processing error: {str(e)}",
                "timestamp": asyncio.get_event_loop().time()
            }
            await self.queue.put(error_chunk)

    async def generate_real_time_stream(self, agent, query: str) -> AsyncGenerator[str, None]:
        try:
            self.is_streaming = True

            yield f"data: {json.dumps({'type': 'start', 'message': 'Agent starting to process your query...'})}\n\n"

            agent_task = asyncio.create_task(
                astream_graph(
                    agent,
                    {"messages": query},
                    callback=self.real_time_callback,
                    show_tool_info=True
                )
            )

            while self.is_streaming:
                try:
                    chunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    yield f"data: {json.dumps(chunk)}\n\n"
                except asyncio.TimeoutError:
                    if agent_task.done():
                        final_result = await agent_task

                        while not self.queue.empty():
                            chunk = await self.queue.get()
                            yield f"data: {json.dumps(chunk)}\n\n"

                        if hasattr(final_result, 'content'):
                            final_content = str(final_result.content)
                        elif isinstance(final_result, dict):
                            final_content = str(final_result.get('content', final_result))
                        else:
                            final_content = str(final_result)

                        yield f"data: {json.dumps({'type': 'final', 'result': final_content, 'complete': True})}\n\n"
                        break

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            self.is_streaming = False


class Mem0StreamingService:
    def __init__(self, memory_service: Mem0ConversationService):
        self.memory_service = memory_service
        self.streamer = RealTimeAgentStreamer()
        self.current_session_id = None
        self.current_user_message = ""
        self.current_agent_response = ""

    async def memory_aware_callback(self, data: Dict[str, Any]):
        await self.streamer.real_time_callback(data)

        content = data.get("content", "")
        if hasattr(content, 'content'):
            if isinstance(content.content, str):
                self.current_agent_response += content.content
            elif isinstance(content.content, list):
                for item in content.content:
                    if isinstance(item, dict) and "text" in item:
                        self.current_agent_response += item["text"]
        elif isinstance(content, str):
            self.current_agent_response += content

    # async def generate_memory_stream(self, agent, message: str, session_id: str) -> AsyncGenerator[str, None]:
    #     self.current_user_message = message
    #     self.current_agent_response = ''
    #     self.current_session_id = session_id
    #
    #
    #     self.streamer.is_streaming = True
    #     self.streamer.queue = asyncio.Queue()
    #
    #     try:
    #         memories_text, conversation = await self.memory_service.chat_with_memory(message, session_id)
    #
    #         if memories_text:
    #             yield f"data: {json.dumps({'type': 'memory', 'content': 'Found relevant memories from previous conversations'})}\n\n"
    #
    #         yield f"data: {json.dumps({'type': 'start', 'message': 'Agent starting to process your query with memory context...'})}\n\n"
    #
    #         agent_task = asyncio.create_task(
    #             astream_graph(
    #                 agent,
    #                 {"messages": conversation},  # Use conversation with memory context
    #                 callback=self.memory_aware_callback,
    #                 show_tool_info=True
    #             )
    #         )
    #
    #         while self.streamer.is_streaming:
    #             try:
    #                 chunk = await asyncio.wait_for(self.streamer.queue.get(), timeout=0.1)
    #                 yield f"data: {json.dumps(chunk)}\n\n"
    #             except asyncio.TimeoutError:
    #                 if agent_task.done():
    #                     final_result = await agent_task
    #
    #                     while not self.streamer.queue.empty():
    #                         chunk = await self.streamer.queue.get()
    #                         yield f"data: {json.dumps(chunk)}\n\n"
    #
    #                     await self.memory_service.save_conversation_memory(
    #                         self.current_user_message,
    #                         self.current_agent_response,
    #                         session_id
    #                     )
    #
    #                     if hasattr(final_result, 'content'):
    #                         final_content = str(final_result.content)
    #                     elif isinstance(final_result, dict):
    #                         final_content = str(final_result.get('content', final_result))
    #                     else:
    #                         final_content = str(final_result)
    #
    #                     yield f"data: {json.dumps({'type': 'final', 'result': final_content, 'complete': True, 'saved_to_memory': True})}\n\n"
    #                     break
    #
    #     except Exception as e:
    #         yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    #     finally:
    #         self.streamer.is_streaming = False

    async def generate_memory_stream(self, agent, message: str, session_id: str) -> AsyncGenerator[str, None]:
        """
        Alternative: Directly reuse the entire streaming logic from RealTimeAgentStreamer
        and add memory functionality on top
        """
        self.current_user_message = message
        self.current_agent_response = ''
        self.current_session_id = session_id

        try:
            memories_text, conversation = await self.memory_service.chat_with_memory(message, session_id)

            if memories_text:
                yield f"data: {json.dumps({'type': 'memory', 'content': 'Using conversation history and context'})}\n\n"

            async for chunk in self.streamer.generate_real_time_stream(agent, conversation):
                if 'final' in chunk and 'complete' in chunk:
                    await self.memory_service.save_conversation_memory(
                        self.current_user_message,
                        self.current_agent_response,
                        session_id
                    )
                    chunk_data = json.loads(chunk.split('data: ')[1].split('\n\n')[0])
                    chunk_data['saved_to_memory'] = True
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                else:
                    yield chunk

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"