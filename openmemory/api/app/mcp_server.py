"""
MCP Server for OpenMemory with resilient memory client handling.

This module implements an MCP (Model Context Protocol) server that provides
memory operations for OpenMemory. The memory client is initialized lazily
to prevent server crashes when external dependencies (like Ollama) are
unavailable. If the memory client cannot be initialized, the server will
continue running with limited functionality and appropriate error messages.

Key features:
- Lazy memory client initialization
- Graceful error handling for unavailable dependencies
- Fallback to database-only mode when vector store is unavailable
- Proper logging for debugging connection issues
- Environment variable parsing for API keys
"""

import contextvars
import datetime
import json
import logging
import uuid

from app.database import SessionLocal
from app.models import Memory, MemoryAccessLog, MemoryState, MemoryStatusHistory
from app.utils.db import get_user_and_app
from app.utils.memory import get_memory_client
from app.utils.client_configs import connection_monitor
from app.utils.permissions import check_memory_access_permissions
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.routing import APIRouter
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from qdrant_client import models as qdrant_models

# Load environment variables
load_dotenv()

# Initialize MCP
mcp = FastMCP("mem0-mcp-server")

# Don't initialize memory client at import time - do it lazily when needed
def get_memory_client_safe():
    """Get memory client with error handling. Returns None if client cannot be initialized."""
    try:
        return get_memory_client()
    except Exception as e:
        logging.warning(f"Failed to get memory client: {e}")
        return None

# Context variables for user_id and client_name
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id")
client_name_var: contextvars.ContextVar[str] = contextvars.ContextVar("client_name")

# Create a router for MCP endpoints
mcp_router = APIRouter(prefix="/mcp")

# Initialize SSE transport
sse = SseServerTransport("/mcp/messages/")

@mcp.tool(description="Add a new memory. This method is called everytime the user informs anything about themselves, their preferences, or anything that has any relevant information which can be useful in the future conversation. This can also be called when the user asks you to remember something.")
async def add_memories(text: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)

    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    
    # Update connection activity
    connection_monitor.update_activity(client_name, uid)

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Check if app is active
            if not app.is_active:
                return f"Error: App {app.name} is currently paused on OpenMemory. Cannot create new memories."

            # Add timeout handling for memory client operations
            import asyncio
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            
            def add_memory_with_timeout():
                return memory_client.add(text,
                                       user_id=uid,
                                       metadata={
                                          "source_app": "openmemory",
                                          "mcp_client": client_name,
                                      })
            
            # Execute with timeout (30 seconds)
            try:
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(add_memory_with_timeout)
                    response = future.result(timeout=30)
            except FutureTimeoutError:
                return json.dumps({
                    "status": "error",
                    "message": "Memory operation timed out. Please try again with shorter text or check your API configuration.",
                    "error_code": "TIMEOUT"
                }, indent=2)
            except Exception as e:
                return json.dumps({
                    "status": "error", 
                    "message": f"Memory operation failed: {str(e)}",
                    "error_code": "API_ERROR"
                }, indent=2)

            # Process the response and update database
            if isinstance(response, dict) and 'results' in response:
                for result in response['results']:
                    memory_id = uuid.UUID(result['id'])
                    memory = db.query(Memory).filter(Memory.id == memory_id).first()

                    if result['event'] == 'ADD':
                        if not memory:
                            memory = Memory(
                                id=memory_id,
                                user_id=user.id,
                                app_id=app.id,
                                content=result['memory'],
                                state=MemoryState.active
                            )
                            db.add(memory)
                        else:
                            memory.state = MemoryState.active
                            memory.content = result['memory']

                        # Create history entry
                        history = MemoryStatusHistory(
                            memory_id=memory_id,
                            changed_by=user.id,
                            old_state=MemoryState.deleted if memory else None,
                            new_state=MemoryState.active
                        )
                        db.add(history)

                    elif result['event'] == 'DELETE':
                        if memory:
                            memory.state = MemoryState.deleted
                            memory.deleted_at = datetime.datetime.now(datetime.UTC)
                            # Create history entry
                            history = MemoryStatusHistory(
                                memory_id=memory_id,
                                changed_by=user.id,
                                old_state=MemoryState.active,
                                new_state=MemoryState.deleted
                            )
                            db.add(history)

                db.commit()

            # Return simplified response instead of full mem0 response
            if isinstance(response, dict) and 'results' in response:
                added_count = len([r for r in response['results'] if r['event'] == 'ADD'])
                updated_count = len([r for r in response['results'] if r['event'] == 'UPDATE'])
                deleted_count = len([r for r in response['results'] if r['event'] == 'DELETE'])
                
                simple_response = {
                    "status": "success",
                    "message": f"Memory operation completed",
                    "added": added_count,
                    "updated": updated_count,
                    "deleted": deleted_count
                }
                return json.dumps(simple_response, indent=2)
            else:
                return json.dumps({"status": "success", "message": "Memory added successfully"}, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error adding to memory: {e}")
        return f"Error adding to memory: {e}"


@mcp.tool(description="Search through stored memories. This method is called EVERYTIME the user asks anything.")
async def search_memory(query: str) -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"
    
    # Update connection activity
    connection_monitor.update_activity(client_name, uid)

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Use mem0's search method which is compatible with all vector stores
            try:
                search_results = memory_client.search(
                    query=query,
                    user_id=uid,
                    limit=10
                )
                
                # Process search results - handle both old and new response formats
                if isinstance(search_results, dict) and 'results' in search_results:
                    memories = search_results['results']
                elif isinstance(search_results, list):
                    memories = search_results
                else:
                    memories = []
                
                # Format memories consistently
                formatted_memories = [
                    {
                        "id": memory.get("id"),
                        "memory": memory.get("memory"),
                        "score": memory.get("score", 0.0),
                        "created_at": memory.get("created_at"),
                        "updated_at": memory.get("updated_at"),
                    }
                    for memory in memories
                ]

                # Log memory access for each memory found
                for memory_data in memories:
                    if memory_data.get('id'):
                        try:
                            memory_id = uuid.UUID(memory_data['id'])
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="search",
                                metadata_={
                                    "query": query,
                                    "score": memory_data.get('score'),
                                }
                            )
                            db.add(access_log)
                        except (ValueError, TypeError) as e:
                            logging.warning(f"Error creating access log for memory {memory_data.get('id')}: {e}")

                db.commit()

                # Return formatted search results
                if formatted_memories:
                    result = f"Found {len(formatted_memories)} relevant memories:\n\n"
                    for i, memory in enumerate(formatted_memories[:5], 1):  # Limit to top 5
                        result += f"{i}. {memory['memory'][:200]}{'...' if len(memory['memory']) > 200 else ''}\n"
                        if memory.get('score'):
                            result += f"   Relevance: {memory['score']:.3f}\n"
                        result += "\n"
                    return result
                else:
                    return f"No memories found related to: {query}"
            
            except Exception as search_error:
                logging.error(f"Error during memory search: {search_error}")
                return f"Error searching memory: {search_error}"

        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error searching memories: {e}")
        return f"Error searching memories: {e}"


@mcp.tool(description="List all memories in the user's memory")
async def list_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            # Get all memories
            memories = memory_client.get_all(user_id=uid)
            filtered_memories = []

            # Filter memories based on permissions
            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]
            if isinstance(memories, dict) and 'results' in memories:
                for memory_data in memories['results']:
                    if 'id' in memory_data:
                        memory_id = uuid.UUID(memory_data['id'])
                        if memory_id in accessible_memory_ids:
                            # Create access log entry
                            access_log = MemoryAccessLog(
                                memory_id=memory_id,
                                app_id=app.id,
                                access_type="list",
                                metadata_={
                                    "hash": memory_data.get('hash')
                                }
                            )
                            db.add(access_log)
                            filtered_memories.append(memory_data)
                db.commit()
            else:
                for memory in memories:
                    memory_id = uuid.UUID(memory['id'])
                    memory_obj = db.query(Memory).filter(Memory.id == memory_id).first()
                    if memory_obj and check_memory_access_permissions(db, memory_obj, app.id):
                        # Create access log entry
                        access_log = MemoryAccessLog(
                            memory_id=memory_id,
                            app_id=app.id,
                            access_type="list",
                            metadata_={
                                "hash": memory.get('hash')
                            }
                        )
                        db.add(access_log)
                        filtered_memories.append(memory)
                db.commit()
            return json.dumps(filtered_memories, indent=2)
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error getting memories: {e}")
        return f"Error getting memories: {e}"


@mcp.tool(description="Delete all memories in the user's memory")
async def delete_all_memories() -> str:
    uid = user_id_var.get(None)
    client_name = client_name_var.get(None)
    if not uid:
        return "Error: user_id not provided"
    if not client_name:
        return "Error: client_name not provided"

    # Get memory client safely
    memory_client = get_memory_client_safe()
    if not memory_client:
        return "Error: Memory system is currently unavailable. Please try again later."

    try:
        db = SessionLocal()
        try:
            # Get or create user and app
            user, app = get_user_and_app(db, user_id=uid, app_id=client_name)

            user_memories = db.query(Memory).filter(Memory.user_id == user.id).all()
            accessible_memory_ids = [memory.id for memory in user_memories if check_memory_access_permissions(db, memory, app.id)]

            # delete the accessible memories only
            for memory_id in accessible_memory_ids:
                try:
                    memory_client.delete(memory_id)
                except Exception as delete_error:
                    logging.warning(f"Failed to delete memory {memory_id} from vector store: {delete_error}")

            # Update each memory's state and create history entries
            now = datetime.datetime.now(datetime.UTC)
            for memory_id in accessible_memory_ids:
                memory = db.query(Memory).filter(Memory.id == memory_id).first()
                # Update memory state
                memory.state = MemoryState.deleted
                memory.deleted_at = now

                # Create history entry
                history = MemoryStatusHistory(
                    memory_id=memory_id,
                    changed_by=user.id,
                    old_state=MemoryState.active,
                    new_state=MemoryState.deleted
                )
                db.add(history)

                # Create access log entry
                access_log = MemoryAccessLog(
                    memory_id=memory_id,
                    app_id=app.id,
                    access_type="delete_all",
                    metadata_={"operation": "bulk_delete"}
                )
                db.add(access_log)

            db.commit()
            return "Successfully deleted all memories"
        finally:
            db.close()
    except Exception as e:
        logging.exception(f"Error deleting memories: {e}")
        return f"Error deleting memories: {e}"


@mcp_router.get("/{client_name}/sse/{user_id}")
async def handle_sse(request: Request):
    """Handle SSE connections for a specific user and client"""
    # Extract user_id and client_name from path parameters
    uid = request.path_params.get("user_id")
    user_token = user_id_var.set(uid or "")
    client_name = request.path_params.get("client_name")
    client_token = client_name_var.set(client_name or "")
    
    # Register client connection
    connection_monitor.register_connection(
        client_name=client_name or "unknown",
        user_id=uid or "default",
        connection_info={
            "user_agent": request.headers.get("user-agent", ""),
            "remote_addr": request.client.host if request.client else "unknown"
        }
    )

    try:
        # Handle SSE connection
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,
        ) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )
    finally:
        # Disconnect client
        connection_monitor.disconnect_client(client_name or "unknown", uid or "default")
        # Clean up context variables
        user_id_var.reset(user_token)
        client_name_var.reset(client_token)


@mcp_router.post("/messages/")
async def handle_get_message(request: Request):
    return await handle_post_message(request)


@mcp_router.post("/{client_name}/sse/{user_id}/messages/")
async def handle_post_message(request: Request):
    return await handle_post_message(request)

# async def handle_post_message(request: Request):
#     """Handle POST messages for SSE"""
#     try:
#         body = await request.body()

#         # Create a simple receive function that returns the body
#         async def receive():
#             return {"type": "http.request", "body": body, "more_body": False}

#         # Create a simple send function that does nothing
#         async def send(message):
#             return {}

#         # Call handle_post_message with the correct arguments
#         await sse.handle_post_message(request.scope, receive, send)

#         # Return a success response
#         return {"status": "ok"}
#     finally:
#         pass
#         # Clean up context variable
#         # client_name_var.reset(client_token)

def setup_mcp_server(app: FastAPI):
    """Setup MCP server with the FastAPI application"""
    mcp._mcp_server.name = "mem0-mcp-server"

    # Include MCP router in the FastAPI app
    app.include_router(mcp_router)
