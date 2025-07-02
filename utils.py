from typing import Any, Dict, List, Callable, Optional
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
import uuid
import json


def random_uuid():
    return str(uuid.uuid4())


def extract_tool_info(content: Any) -> Dict[str, Any]:
    """Extract tool information from message content."""
    tool_info = {
        "tool_calls": [],
        "tool_results": [],
        "has_tools": False
    }
    
    if isinstance(content, BaseMessage):
        # Check for tool calls in AI messages
        if hasattr(content, 'tool_calls') and content.tool_calls:
            tool_info["has_tools"] = True
            for tool_call in content.tool_calls:
                tool_info["tool_calls"].append({
                    "id": getattr(tool_call, 'id', None),
                    "name": getattr(tool_call, 'name', None),
                    "args": getattr(tool_call, 'args', {}),
                    "type": getattr(tool_call, 'type', 'function')
                })
        
        # Check for tool results in tool messages
        if isinstance(content, ToolMessage):
            tool_info["has_tools"] = True
            tool_info["tool_results"].append({
                "tool_call_id": getattr(content, 'tool_call_id', None),
                "content": content.content,
                "name": getattr(content, 'name', None)
            })
    
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, BaseMessage):
                sub_info = extract_tool_info(item)
                if sub_info["has_tools"]:
                    tool_info["has_tools"] = True
                    tool_info["tool_calls"].extend(sub_info["tool_calls"])
                    tool_info["tool_results"].extend(sub_info["tool_results"])
    
    return tool_info


def print_tool_info(tool_info: Dict[str, Any], node_name: str):
    """Print formatted tool information."""
    if not tool_info["has_tools"]:
        return
    
    print(f"\n🔧 \033[1;35mTool Activity in {node_name}\033[0m 🔧")
    
    # Print tool calls
    for tool_call in tool_info["tool_calls"]:
        print(f"  📞 Calling: \033[1;33m{tool_call['name']}\033[0m")
        if tool_call['id']:
            print(f"     ID: {tool_call['id']}")
        if tool_call['args']:
            print(f"     Args: {json.dumps(tool_call['args'], indent=6)}")
    
    # Print tool results
    for tool_result in tool_info["tool_results"]:
        print(f"  📋 Result from: \033[1;32m{tool_result['name'] or 'Unknown Tool'}\033[0m")
        if tool_result['tool_call_id']:
            print(f"     Call ID: {tool_result['tool_call_id']}")
        if tool_result['content']:
            # Truncate long content
            content_str = str(tool_result['content'])
            if len(content_str) > 200:
                content_str = content_str[:200] + "..."
            print(f"     Content: {content_str}")


async def astream_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    stream_mode: str = "messages",
    include_subgraphs: bool = False,
    show_tool_info: bool = True,  # New parameter
    show_metadata: bool = False,  # New parameter
) -> Dict[str, Any]:
    """
    Asynchronously streams and directly outputs LangGraph execution results with enhanced tool information.

    Args:
        graph (CompiledStateGraph): Compiled LangGraph object to execute
        inputs (dict): Input dictionary to pass to the graph
        config (Optional[RunnableConfig]): Execution configuration (optional)
        node_names (List[str], optional): List of node names to output. Default is empty list
        callback (Optional[Callable], optional): Callback function for processing each chunk. Default is None
            The callback function receives a dictionary in the form {"node": str, "content": Any, "tool_info": Dict}.
        stream_mode (str, optional): Streaming mode ("messages" or "updates"). Default is "messages"
        include_subgraphs (bool, optional): Whether to include subgraphs. Default is False
        show_tool_info (bool, optional): Whether to display tool information. Default is True
        show_metadata (bool, optional): Whether to display metadata information. Default is False

    Returns:
        Dict[str, Any]: Final result with enhanced information
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    prev_node = ""

    if stream_mode == "messages":
        async for chunk_msg, metadata in graph.astream(
            inputs, config, stream_mode=stream_mode
        ):
            curr_node = metadata["langgraph_node"]
            
            # Extract tool information
            tool_info = extract_tool_info(chunk_msg) if show_tool_info else {}
            
            final_result = {
                "node": curr_node,
                "content": chunk_msg,
                "metadata": metadata,
                "tool_info": tool_info,
            }

            # Process only if node_names is empty or current node is in node_names
            if not node_names or curr_node in node_names:
                # Execute callback function if available
                if callback:
                    result = callback({
                        "node": curr_node, 
                        "content": chunk_msg,
                        "tool_info": tool_info,
                        "metadata": metadata
                    })
                    if hasattr(result, "__await__"):
                        await result
                # Default output if no callback
                else:
                    # Output separator only when node changes
                    if curr_node != prev_node:
                        print("\n" + "=" * 50)
                        print(f"🔄 Node: \033[1;36m{curr_node}\033[0m 🔄")
                        print("- " * 25)
                        
                        # Show metadata if requested
                        if show_metadata:
                            print(f"📊 Metadata: {json.dumps(metadata, indent=2)}")
                    
                    # Show tool information
                    if show_tool_info:
                        print_tool_info(tool_info, curr_node)

                    # Handle Claude/Anthropic model token chunks - always extract text only
                    if hasattr(chunk_msg, "content"):
                        # List-type content (Anthropic/Claude style)
                        if isinstance(chunk_msg.content, list):
                            for item in chunk_msg.content:
                                if isinstance(item, dict) and "text" in item:
                                    print(item["text"], end="", flush=True)
                        # String-type content
                        elif isinstance(chunk_msg.content, str):
                            print(chunk_msg.content, end="", flush=True)
                    # Handle other types of chunk_msg
                    else:
                        print(chunk_msg, end="", flush=True)

                prev_node = curr_node

    elif stream_mode == "updates":
        async for chunk in graph.astream(
            inputs, config, stream_mode=stream_mode, subgraphs=include_subgraphs
        ):
            # Branch processing method based on return format
            if isinstance(chunk, tuple) and len(chunk) == 2:
                namespace, node_chunks = chunk
            else:
                namespace = []
                node_chunks = chunk

            # Check if it's a dictionary and process items
            if isinstance(node_chunks, dict):
                for node_name, node_chunk in node_chunks.items():
                    # Extract tool information from node chunk
                    tool_info = {}
                    if show_tool_info:
                        if isinstance(node_chunk, dict):
                            for k, v in node_chunk.items():
                                chunk_tool_info = extract_tool_info(v)
                                if chunk_tool_info["has_tools"]:
                                    if not tool_info.get("has_tools"):
                                        tool_info = {"tool_calls": [], "tool_results": [], "has_tools": False}
                                    tool_info["has_tools"] = True
                                    tool_info["tool_calls"].extend(chunk_tool_info["tool_calls"])
                                    tool_info["tool_results"].extend(chunk_tool_info["tool_results"])
                        else:
                            tool_info = extract_tool_info(node_chunk)
                    
                    final_result = {
                        "node": node_name,
                        "content": node_chunk,
                        "namespace": namespace,
                        "tool_info": tool_info,
                    }

                    # Filter only if node_names is not empty
                    if len(node_names) > 0 and node_name not in node_names:
                        continue

                    # Execute callback function if available
                    if callback is not None:
                        result = callback({
                            "node": node_name, 
                            "content": node_chunk,
                            "tool_info": tool_info,
                            "namespace": namespace
                        })
                        if hasattr(result, "__await__"):
                            await result
                    # Default output if no callback
                    else:
                        # Output separator only when node changes
                        if node_name != prev_node:
                            print("\n" + "=" * 50)
                            print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                            print("- " * 25)
                        
                        # Show tool information
                        if show_tool_info:
                            print_tool_info(tool_info, node_name)

                        # Output node chunk data - process with text focus
                        if isinstance(node_chunk, dict):
                            for k, v in node_chunk.items():
                                if isinstance(v, BaseMessage):
                                    # Handle cases where BaseMessage's content attribute is text or list
                                    if hasattr(v, "content"):
                                        if isinstance(v.content, list):
                                            for item in v.content:
                                                if (
                                                    isinstance(item, dict)
                                                    and "text" in item
                                                ):
                                                    print(
                                                        item["text"], end="", flush=True
                                                    )
                                        else:
                                            print(v.content, end="", flush=True)
                                    else:
                                        v.pretty_print()
                                elif isinstance(v, list):
                                    for list_item in v:
                                        if isinstance(list_item, BaseMessage):
                                            if hasattr(list_item, "content"):
                                                if isinstance(list_item.content, list):
                                                    for item in list_item.content:
                                                        if (
                                                            isinstance(item, dict)
                                                            and "text" in item
                                                        ):
                                                            print(
                                                                item["text"],
                                                                end="",
                                                                flush=True,
                                                            )
                                                else:
                                                    print(
                                                        list_item.content,
                                                        end="",
                                                        flush=True,
                                                    )
                                            else:
                                                list_item.pretty_print()
                                        elif (
                                            isinstance(list_item, dict)
                                            and "text" in list_item
                                        ):
                                            print(list_item["text"], end="", flush=True)
                                        else:
                                            print(list_item, end="", flush=True)
                                elif isinstance(v, dict) and "text" in v:
                                    print(v["text"], end="", flush=True)
                                else:
                                    print(v, end="", flush=True)
                        elif node_chunk is not None:
                            if hasattr(node_chunk, "__iter__") and not isinstance(
                                node_chunk, str
                            ):
                                for item in node_chunk:
                                    if isinstance(item, dict) and "text" in item:
                                        print(item["text"], end="", flush=True)
                                    else:
                                        print(item, end="", flush=True)
                            else:
                                print(node_chunk, end="", flush=True)

                    prev_node = node_name
            else:
                print("\n" + "=" * 50)
                print(f"🔄 Raw output 🔄")
                print("- " * 25)
                print(node_chunks, end="", flush=True)
                final_result = {"content": node_chunks}

    else:
        raise ValueError(
            f"Invalid stream_mode: {stream_mode}. Must be 'messages' or 'updates'."
        )

    return final_result


# Example usage with custom callback to capture tool information
async def tool_tracking_callback(data: Dict[str, Any]):
    """Example callback that tracks tool usage."""
    node = data["node"]
    tool_info = data.get("tool_info", {})
    
    if tool_info.get("has_tools"):
        print(f"\n📊 TOOL TRACKER: Node '{node}' is using tools!")
        for tool_call in tool_info.get("tool_calls", []):
            print(f"  - Tool: {tool_call['name']} with args: {tool_call['args']}")
        for tool_result in tool_info.get("tool_results", []):
            print(f"  - Got result from: {tool_result['name']}")


# Enhanced ainvoke_graph with tool information
async def ainvoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: Optional[RunnableConfig] = None,
    node_names: List[str] = [],
    callback: Optional[Callable] = None,
    include_subgraphs: bool = True,
    show_tool_info: bool = True,
    show_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Enhanced ainvoke_graph with tool information tracking.
    """
    config = config or {}
    final_result = {}

    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"

    async for chunk in graph.astream(
        inputs, config, stream_mode="updates", subgraphs=include_subgraphs
    ):
        if isinstance(chunk, tuple) and len(chunk) == 2:
            namespace, node_chunks = chunk
        else:
            namespace = []
            node_chunks = chunk

        if isinstance(node_chunks, dict):
            for node_name, node_chunk in node_chunks.items():
                # Extract tool information
                tool_info = {}
                if show_tool_info:
                    if isinstance(node_chunk, dict):
                        for k, v in node_chunk.items():
                            chunk_tool_info = extract_tool_info(v)
                            if chunk_tool_info["has_tools"]:
                                if not tool_info.get("has_tools"):
                                    tool_info = {"tool_calls": [], "tool_results": [], "has_tools": False}
                                tool_info["has_tools"] = True
                                tool_info["tool_calls"].extend(chunk_tool_info["tool_calls"])
                                tool_info["tool_results"].extend(chunk_tool_info["tool_results"])
                    else:
                        tool_info = extract_tool_info(node_chunk)
                
                final_result = {
                    "node": node_name,
                    "content": node_chunk,
                    "namespace": namespace,
                    "tool_info": tool_info,
                }

                if node_names and node_name not in node_names:
                    continue

                if callback is not None:
                    result = callback({
                        "node": node_name, 
                        "content": node_chunk,
                        "tool_info": tool_info,
                        "namespace": namespace
                    })
                    if hasattr(result, "__await__"):
                        await result
                else:
                    print("\n" + "=" * 50)
                    formatted_namespace = format_namespace(namespace)
                    if formatted_namespace == "root graph":
                        print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                    else:
                        print(
                            f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                        )
                    print("- " * 25)
                    
                    # Show tool information
                    if show_tool_info:
                        print_tool_info(tool_info, node_name)
                    
                    # Show metadata if requested
                    if show_metadata and "metadata" in final_result:
                        print(f"📊 Metadata: {json.dumps(final_result['metadata'], indent=2)}")

                    # Output node chunk data
                    if isinstance(node_chunk, dict):
                        for k, v in node_chunk.items():
                            if isinstance(v, BaseMessage):
                                v.pretty_print()
                            elif isinstance(v, list):
                                for list_item in v:
                                    if isinstance(list_item, BaseMessage):
                                        list_item.pretty_print()
                                    else:
                                        print(list_item)
                            elif isinstance(v, dict):
                                for node_chunk_key, node_chunk_value in v.items():
                                    print(f"{node_chunk_key}:\n{node_chunk_value}")
                            else:
                                print(f"\033[1;32m{k}\033[0m:\n{v}")
                    elif node_chunk is not None:
                        if hasattr(node_chunk, "__iter__") and not isinstance(
                            node_chunk, str
                        ):
                            for item in node_chunk:
                                print(item)
                        else:
                            print(node_chunk)
                    print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print(f"🔄 Raw output 🔄")
            print("- " * 25)
            print(node_chunks)
            print("=" * 50)
            final_result = {"content": node_chunks}

    return final_result