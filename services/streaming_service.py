from typing import Dict, Any, AsyncGenerator
import asyncio

import json

from utils import astream_graph



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