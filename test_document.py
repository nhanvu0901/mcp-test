import asyncio
import json
from pathlib import Path
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


async def test_document_processor():
    """Test the document processor MCP tool"""
    
    # Connect to your DocumentService MCP server running on localhost:8001
    async with sse_client("http://localhost:8001/sse") as (read_stream, write_stream):
        # Create a client session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            print("✅ Connected to DocumentService MCP server")
            
            # List available tools
            tools_response = await session.list_tools()
            print(f"📋 Available tools: {[tool.name for tool in tools_response.tools]}")
            
            # Test case
            test_case = {
                "name": "Test Markdown document",
                "file_path": "data/mcp.md",
                "filename": "mcp.md",
                "document_id": "doc_001"
            }
            
            print(f"\n🧪 Test: {test_case['name']}")
            print(f"   File: {test_case['file_path']}")
            
            try:
                # Call the process_document tool
                result = await session.call_tool(
                    "process_document",
                    arguments={
                        "file_path": test_case["file_path"],
                        "filename": test_case["filename"],
                        "document_id": test_case["document_id"]
                    }
                )
                
                # Parse and display result
                if result.content:
                    # Extract text content from the result
                    content_text = result.content[0].text if result.content else "No content"
                    try:
                        result_dict = json.loads(content_text)
                        print(f"   ✅ Status: {result_dict.get('status', 'unknown')}")
                        
                        if result_dict.get('status') == 'success':
                            print(f"   📄 Document ID: {result_dict.get('document_id')}")
                            print(f"   📁 Collection: {result_dict.get('collection_id')}")
                            print(f"   📝 Filename: {result_dict.get('filename')}")
                        else:
                            print(f"   ❌ Error: {result_dict.get('error', 'Unknown error')}")
                                 
                    except json.JSONDecodeError:
                        print(f"   📄 Raw response: {content_text}")
                else:
                    print("   ⚠️  No response content")
                        
            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
            
            print(f"\n🎉 Testing completed!")


if __name__ == "__main__":
    # Run the test
    print("🚀 Starting Document Processor MCP Tool Test")
    asyncio.run(test_document_processor())