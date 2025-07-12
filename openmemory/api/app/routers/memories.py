from datetime import datetime, timezone
from typing import List, Optional, Set, Dict
from uuid import UUID, uuid4
import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from fastapi_pagination import Page, Params
from fastapi_pagination.ext.sqlalchemy import paginate as sqlalchemy_paginate
from pydantic import BaseModel
from sqlalchemy import or_, func
from app.utils.memory import get_memory_client
import json

from app.database import get_db
from app.models import (
    AccessControl,
    App,
    Category,
    Memory,
    MemoryAccessLog,
    MemoryState,
    MemoryStatusHistory,
    User,
)
from app.schemas import MemoryResponse
from app.utils.memory import get_memory_client
from app.utils.permissions import check_memory_access_permissions

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])


def get_memory_or_404(db: Session, memory_id: UUID) -> Memory:
    memory = db.query(Memory).filter(Memory.id == memory_id).first()
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


def update_memory_state(db: Session, memory_id: UUID, new_state: MemoryState, user_id: UUID):
    memory = get_memory_or_404(db, memory_id)
    old_state = memory.state

    # Update memory state
    memory.state = new_state
    if new_state == MemoryState.archived:
        memory.archived_at = datetime.now(timezone.utc)
    elif new_state == MemoryState.deleted:
        memory.deleted_at = datetime.now(timezone.utc)

    # Record state change
    history = MemoryStatusHistory(
        memory_id=memory_id,
        changed_by=user_id,
        old_state=old_state,
        new_state=new_state
    )
    db.add(history)
    db.commit()
    return memory


def get_accessible_memory_ids(db: Session, app_id: UUID) -> Set[UUID]:
    """
    Get the set of memory IDs that the app has access to based on app-level ACL rules.
    Returns all memory IDs if no specific restrictions are found.
    """
    # Get app-level access controls
    app_access = db.query(AccessControl).filter(
        AccessControl.subject_type == "app",
        AccessControl.subject_id == app_id,
        AccessControl.object_type == "memory"
    ).all()

    # If no app-level rules exist, return None to indicate all memories are accessible
    if not app_access:
        return None

    # Initialize sets for allowed and denied memory IDs
    allowed_memory_ids = set()
    denied_memory_ids = set()

    # Process app-level rules
    for rule in app_access:
        if rule.effect == "allow":
            if rule.object_id:  # Specific memory access
                allowed_memory_ids.add(rule.object_id)
            else:  # All memories access
                return None  # All memories allowed
        elif rule.effect == "deny":
            if rule.object_id:  # Specific memory denied
                denied_memory_ids.add(rule.object_id)
            else:  # All memories denied
                return set()  # No memories accessible

    # Remove denied memories from allowed set
    if allowed_memory_ids:
        allowed_memory_ids -= denied_memory_ids

    return allowed_memory_ids


# List all memories with filtering
@router.get("/", response_model=Page[MemoryResponse])
async def list_memories(
    user_id: str,
    app_id: Optional[UUID] = None,
    from_date: Optional[int] = Query(
        None,
        description="Filter memories created after this date (timestamp)",
        examples=[1718505600]
    ),
    to_date: Optional[int] = Query(
        None,
        description="Filter memories created before this date (timestamp)",
        examples=[1718505600]
    ),
    categories: Optional[str] = None,
    params: Params = Depends(),
    search_query: Optional[str] = None,
    sort_column: Optional[str] = Query(None, description="Column to sort by (memory, categories, app_name, created_at)"),
    sort_direction: Optional[str] = Query(None, description="Sort direction (asc or desc)"),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query with joins
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
        Memory.state != MemoryState.archived,
        Memory.content.ilike(f"%{search_query}%") if search_query else True
    ).join(App, Memory.app_id == App.id)

    # Apply filters
    if app_id:
        query = query.filter(Memory.app_id == app_id)

    if from_date:
        from_datetime = datetime.fromtimestamp(from_date, tz=timezone.utc)
        query = query.filter(Memory.created_at >= from_datetime)

    if to_date:
        to_datetime = datetime.fromtimestamp(to_date, tz=timezone.utc)
        query = query.filter(Memory.created_at <= to_datetime)

    # Apply category filter if provided
    if categories:
        category_list = [c.strip() for c in categories.split(",")]
        query = query.join(Memory.categories).filter(Category.name.in_(category_list))

    # Add eager loading for categories first
    query = query.options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    )

    # Apply sorting - must be compatible with distinct
    if sort_column:
        if sort_column == "app_name":
            sort_field = App.name
        else:
            sort_field = getattr(Memory, sort_column, None)
        if sort_field:
            if sort_direction == "desc":
                query = query.order_by(Memory.id, sort_field.desc())
            else:
                query = query.order_by(Memory.id, sort_field.asc())
    else:
        # Default sorting - Memory.id must come first for distinct
        query = query.order_by(Memory.id, Memory.created_at.desc())

    # Apply distinct after order by is properly set
    query = query.distinct(Memory.id)

    # Use fastapi-pagination's paginate function with transformer
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=json.loads(memory.metadata_) if isinstance(memory.metadata_, str) and memory.metadata_ else (memory.metadata_ or {})
            )
            for memory in items
            if check_memory_access_permissions(db, memory, app_id)
        ]
    )


# Get all categories
@router.get("/categories")
async def get_categories(
    user_id: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get unique categories associated with the user's memories
    # Get all memories
    memories = db.query(Memory).filter(Memory.user_id == user.id, Memory.state != MemoryState.deleted, Memory.state != MemoryState.archived).all()
    # Get all categories from memories
    categories = [category for memory in memories for category in memory.categories]
    # Get unique categories
    unique_categories = list(set(categories))

    return {
        "categories": unique_categories,
        "total": len(unique_categories)
    }


class CreateMemoryRequest(BaseModel):
    user_id: str
    text: str
    metadata: dict = {}
    infer: bool = True
    app: str = "openmemory"
    custom_categories: Optional[List[Dict[str, str]]] = None


# Create new memory
@router.post("/")
async def create_memory(
    request: CreateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    # Get or create app
    app_obj = db.query(App).filter(App.name == request.app,
                                   App.owner_id == user.id).first()
    if not app_obj:
        app_obj = App(name=request.app, owner_id=user.id)
        db.add(app_obj)
        db.commit()
        db.refresh(app_obj)

    # Check if app is active
    if not app_obj.is_active:
        raise HTTPException(status_code=403, detail=f"App {request.app} is currently paused on OpenMemory. Cannot create new memories.")

    # Log what we're about to do (减少详细信息)
    logging.debug(f"Creating memory for user_id: {request.user_id} with app: {request.app}")
    
    # Try to get memory client safely
    try:
        # Get custom categories from config if not provided in request
        db_custom_categories = None
        if not request.custom_categories:
            try:
                from app.models import Config as ConfigModel
                db_config = db.query(ConfigModel).filter(ConfigModel.key == "main").first()
                if db_config and "openmemory" in db_config.value and "custom_categories" in db_config.value["openmemory"]:
                    db_custom_categories = db_config.value["openmemory"]["custom_categories"]
            except Exception as config_error:
                logging.warning(f"Failed to load custom categories from config: {config_error}")
        
        # Use request categories or fall back to config categories
        categories_to_use = request.custom_categories or db_custom_categories
        
        memory_client = get_memory_client()
        if not memory_client:
            raise Exception("Memory client is not available")
    except Exception as client_error:
        logging.warning(f"Memory client unavailable: {client_error}. Creating memory in database only.")
        # Return a json response with the error
        return {
            "error": str(client_error)
        }

    # Try to save to Qdrant via memory_client
    try:
        # Add json keyword to ensure compatibility with third-party OpenAI API providers
        # that require the word "json" in messages when using json_object response format
        enhanced_text = f"{request.text} (store as json format)"
        
        qdrant_response = memory_client.add(
            enhanced_text,
            user_id=request.user_id,  # Use string user_id to match search
            metadata={
                "source_app": "openmemory",
                "mcp_client": request.app,
                "original_text": request.text,  # Store original text for reference
            }
            # Note: custom_categories parameter removed for compatibility with mem0ai 0.1.114
        )
        
        # Log the response for debugging
        logging.debug(f"Memory client response: {len(qdrant_response.get('results', []))} results")
        
        # Process Qdrant response
        if isinstance(qdrant_response, dict) and 'results' in qdrant_response:
            for result in qdrant_response['results']:
                if result['event'] == 'ADD':
                    # Get the Qdrant-generated ID
                    memory_id = UUID(result['id'])
                    
                    # Check if memory already exists
                    existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                    
                    if existing_memory:
                        # Update existing memory
                        existing_memory.state = MemoryState.active
                        existing_memory.content = result['memory']
                        memory = existing_memory
                    else:
                        # Create memory with the EXACT SAME ID from Qdrant
                        memory = Memory(
                            id=memory_id,  # Use the same ID that Qdrant generated
                            user_id=user.id,
                            app_id=app_obj.id,
                            content=result['memory'],
                            metadata_=request.metadata,
                            state=MemoryState.active
                        )
                        db.add(memory)
                    
                    # Create history entry
                    history = MemoryStatusHistory(
                        memory_id=memory_id,
                        changed_by=user.id,
                        old_state=MemoryState.deleted if existing_memory else MemoryState.deleted,
                        new_state=MemoryState.active
                    )
                    db.add(history)
                    
                    db.commit()
                    db.refresh(memory)
                    return memory
    except Exception as qdrant_error:
        logging.warning(f"Qdrant operation failed: {qdrant_error}.")
        # Return a json response with the error
        return {
            "error": str(qdrant_error)
        }




# Get memory by ID
@router.get("/{memory_id}")
async def get_memory(
    memory_id: UUID,
    db: Session = Depends(get_db)
):
    memory = get_memory_or_404(db, memory_id)
    return {
        "id": memory.id,
        "text": memory.content,
        "created_at": int(memory.created_at.timestamp()),
        "state": memory.state.value,
        "app_id": memory.app_id,
        "app_name": memory.app.name if memory.app else None,
        "categories": [category.name for category in memory.categories],
        "metadata_": memory.metadata_
    }


class DeleteMemoriesRequest(BaseModel):
    memory_ids: List[UUID]
    user_id: str

# Delete multiple memories
@router.delete("/")
async def delete_memories(
    request: DeleteMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    for memory_id in request.memory_ids:
        update_memory_state(db, memory_id, MemoryState.deleted, user.id)
    return {"message": f"Successfully deleted {len(request.memory_ids)} memories"}


# Archive memories
@router.post("/actions/archive")
async def archive_memories(
    memory_ids: List[UUID],
    user_id: UUID,
    db: Session = Depends(get_db)
):
    for memory_id in memory_ids:
        update_memory_state(db, memory_id, MemoryState.archived, user_id)
    return {"message": f"Successfully archived {len(memory_ids)} memories"}


class PauseMemoriesRequest(BaseModel):
    memory_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    app_id: Optional[UUID] = None
    all_for_app: bool = False
    global_pause: bool = False
    state: Optional[MemoryState] = None
    user_id: str

# Pause access to memories
@router.post("/actions/pause")
async def pause_memories(
    request: PauseMemoriesRequest,
    db: Session = Depends(get_db)
):
    
    global_pause = request.global_pause
    all_for_app = request.all_for_app
    app_id = request.app_id
    memory_ids = request.memory_ids
    category_ids = request.category_ids
    state = request.state or MemoryState.paused

    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_id = user.id
    
    if global_pause:
        # Pause all memories
        memories = db.query(Memory).filter(
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if app_id:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.app_id == app_id,
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused all memories for app {app_id}"}
    
    if all_for_app and memory_ids:
        # Pause all memories for an app
        memories = db.query(Memory).filter(
            Memory.user_id == user.id,
            Memory.state != MemoryState.deleted,
            Memory.id.in_(memory_ids)
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": "Successfully paused all memories"}

    if memory_ids:
        # Pause specific memories
        for memory_id in memory_ids:
            update_memory_state(db, memory_id, state, user_id)
        return {"message": f"Successfully paused {len(memory_ids)} memories"}

    if category_ids:
        # Pause memories by category
        memories = db.query(Memory).join(Memory.categories).filter(
            Category.id.in_(category_ids),
            Memory.state != MemoryState.deleted,
            Memory.state != MemoryState.archived
        ).all()
        for memory in memories:
            update_memory_state(db, memory.id, state, user_id)
        return {"message": f"Successfully paused memories in {len(category_ids)} categories"}

    raise HTTPException(status_code=400, detail="Invalid pause request parameters")


# Get memory access logs
@router.get("/{memory_id}/access-log")
async def get_memory_access_log(
    memory_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(MemoryAccessLog).filter(MemoryAccessLog.memory_id == memory_id)
    total = query.count()
    logs = query.order_by(MemoryAccessLog.accessed_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    # Get app name
    for log in logs:
        app = db.query(App).filter(App.id == log.app_id).first()
        log.app_name = app.name if app else None

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "logs": logs
    }


class UpdateMemoryRequest(BaseModel):
    memory_content: str
    user_id: str

# Update a memory
@router.put("/{memory_id}")
async def update_memory(
    memory_id: UUID,
    request: UpdateMemoryRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory = get_memory_or_404(db, memory_id)
    memory.content = request.memory_content
    db.commit()
    db.refresh(memory)
    return memory

class FilterMemoriesRequest(BaseModel):
    user_id: str
    page: int = 1
    size: int = 10
    search_query: Optional[str] = None
    app_ids: Optional[List[UUID]] = None
    category_ids: Optional[List[UUID]] = None
    sort_column: Optional[str] = None
    sort_direction: Optional[str] = None
    from_date: Optional[int] = None
    to_date: Optional[int] = None
    show_archived: Optional[bool] = False

@router.post("/filter", response_model=Page[MemoryResponse])
async def filter_memories(
    request: FilterMemoriesRequest,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Build base query
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.state != MemoryState.deleted,
    )

    # Filter archived memories based on show_archived parameter
    if not request.show_archived:
        query = query.filter(Memory.state != MemoryState.archived)

    # Apply search filter
    if request.search_query:
        query = query.filter(Memory.content.ilike(f"%{request.search_query}%"))

    # Apply app filter
    if request.app_ids:
        query = query.filter(Memory.app_id.in_(request.app_ids))

    # Add joins for app and categories
    query = query.outerjoin(App, Memory.app_id == App.id)

    # Apply category filter
    if request.category_ids:
        query = query.join(Memory.categories).filter(Category.id.in_(request.category_ids))
    else:
        query = query.outerjoin(Memory.categories)

    # Apply date filters
    if request.from_date:
        from_datetime = datetime.fromtimestamp(request.from_date, tz=timezone.utc)
        query = query.filter(Memory.created_at >= from_datetime)

    if request.to_date:
        to_datetime = datetime.fromtimestamp(request.to_date, tz=timezone.utc)
        query = query.filter(Memory.created_at <= to_datetime)

    # Add eager loading for categories first
    query = query.options(
        joinedload(Memory.categories)
    )

    # Apply sorting - must be compatible with distinct
    if request.sort_column and request.sort_direction:
        sort_direction = request.sort_direction.lower()
        if sort_direction not in ['asc', 'desc']:
            raise HTTPException(status_code=400, detail="Invalid sort direction")

        sort_mapping = {
            'memory': Memory.content,
            'app_name': App.name,
            'created_at': Memory.created_at
        }

        if request.sort_column not in sort_mapping:
            raise HTTPException(status_code=400, detail="Invalid sort column")

        sort_field = sort_mapping[request.sort_column]
        if sort_direction == 'desc':
            query = query.order_by(Memory.id, sort_field.desc())
        else:
            query = query.order_by(Memory.id, sort_field.asc())
    else:
        # Default sorting - Memory.id must come first for distinct
        query = query.order_by(Memory.id, Memory.created_at.desc())

    # Apply distinct after order by is properly set
    query = query.distinct(Memory.id)

    # Use fastapi-pagination's paginate function
    return sqlalchemy_paginate(
        query,
        Params(page=request.page, size=request.size),
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=json.loads(memory.metadata_) if isinstance(memory.metadata_, str) and memory.metadata_ else (memory.metadata_ or {})
            )
            for memory in items
        ]
    )


class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 10
    threshold: Optional[float] = None


# Search memories using mem0 core
@router.post("/search")
async def search_memories(
    request: SearchMemoryRequest,
    db: Session = Depends(get_db)
):
    """
    Search memories using mem0 core search functionality.
    Returns results in the official format with Score and Categories.
    """
    user = db.query(User).filter(User.user_id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Try to get memory client
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client is not available")
    except Exception as client_error:
        raise HTTPException(status_code=503, detail=f"Memory client error: {str(client_error)}")
    
    try:
        # Search using mem0 core
        search_results = memory_client.search(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit,
            threshold=request.threshold
        )
        
        # Format results in official format
        if isinstance(search_results, dict) and 'results' in search_results:
            formatted_results = []
            for result in search_results['results']:
                formatted_result = {
                    "id": result.get("id"),
                    "memory": result.get("memory"),
                    "score": result.get("score", 0.0),
                    "categories": result.get("categories", []),
                    "created_at": result.get("created_at"),
                    "updated_at": result.get("updated_at"),
                    "metadata": result.get("metadata", {})
                }
                formatted_results.append(formatted_result)
            
            return {
                "results": formatted_results,
                "total": len(formatted_results),
                "query": request.query
            }
        else:
            # Handle legacy format
            if isinstance(search_results, list):
                formatted_results = []
                for result in search_results:
                    formatted_result = {
                        "id": result.get("id"),
                        "memory": result.get("memory"),
                        "score": result.get("score", 0.0),
                        "categories": result.get("categories", []),
                        "created_at": result.get("created_at"),
                        "updated_at": result.get("updated_at"),
                        "metadata": result.get("metadata", {})
                    }
                    formatted_results.append(formatted_result)
                
                return {
                    "results": formatted_results,
                    "total": len(formatted_results),
                    "query": request.query
                }
            else:
                return {
                    "results": [],
                    "total": 0,
                    "query": request.query
                }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/sync")
async def sync_memories_from_vector_store(
    user_id: str,
    force_sync: bool = False,
    db: Session = Depends(get_db)
):
    """
    从mem0向量存储同步记忆到OpenMemory数据库
    这将确保UI能显示所有通过mem0核心创建的记忆
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 获取memory client
    try:
        memory_client = get_memory_client()
        if not memory_client:
            raise HTTPException(status_code=503, detail="Memory client is not available")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory client error: {str(e)}")
    
    try:
        # 从mem0获取所有记忆
        vector_memories_response = memory_client.get_all(user_id=user_id, limit=1000)
        
        # 处理不同的响应格式
        if isinstance(vector_memories_response, dict) and 'results' in vector_memories_response:
            vector_memories = vector_memories_response['results']
        elif isinstance(vector_memories_response, list):
            vector_memories = vector_memories_response
        else:
            vector_memories = []
        
        synced_count = 0
        updated_count = 0
        
        # 获取或创建默认app
        default_app = db.query(App).filter(
            App.name == "mem0_core",
            App.owner_id == user.id
        ).first()
        
        if not default_app:
            default_app = App(
                name="mem0_core",
                owner_id=user.id,
                description="Memories created through mem0 core API"
            )
            db.add(default_app)
            db.commit()
            db.refresh(default_app)
        
        for vector_memory in vector_memories:
            try:
                memory_id = UUID(vector_memory.get('id'))
                memory_content = vector_memory.get('memory', '')
                created_at_str = vector_memory.get('created_at')
                updated_at_str = vector_memory.get('updated_at')
                categories = vector_memory.get('categories', [])
                
                # 检查数据库中是否已存在
                existing_memory = db.query(Memory).filter(Memory.id == memory_id).first()
                
                if existing_memory:
                    # 如果强制同步或内容不同，则更新
                    if force_sync or existing_memory.content != memory_content:
                        existing_memory.content = memory_content
                        if updated_at_str:
                            try:
                                existing_memory.updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                            except:
                                pass
                        
                        # 更新分类
                        if categories:
                            # 清除现有分类
                            existing_memory.categories.clear()
                            # 添加新分类
                            for category_name in categories:
                                category = db.query(Category).filter(
                                    Category.name == category_name,
                                    Category.user_id == user.id
                                ).first()
                                if not category:
                                    category = Category(name=category_name, user_id=user.id)
                                    db.add(category)
                                    db.commit()
                                    db.refresh(category)
                                existing_memory.categories.append(category)
                        
                        updated_count += 1
                else:
                    # 创建新记忆记录
                    new_memory = Memory(
                        id=memory_id,
                        user_id=user.id,
                        app_id=default_app.id,
                        content=memory_content,
                        state=MemoryState.active,
                        metadata_=vector_memory.get('metadata', {})
                    )
                    
                    # 设置时间戳
                    if created_at_str:
                        try:
                            new_memory.created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    if updated_at_str:
                        try:
                            new_memory.updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
                        except:
                            pass
                    
                    db.add(new_memory)
                    db.commit()
                    db.refresh(new_memory)
                    
                    # 添加分类
                    if categories:
                        for category_name in categories:
                            category = db.query(Category).filter(
                                Category.name == category_name,
                                Category.user_id == user.id
                            ).first()
                            if not category:
                                category = Category(name=category_name, user_id=user.id)
                                db.add(category)
                                db.commit()
                                db.refresh(category)
                            new_memory.categories.append(category)
                    
                    synced_count += 1
                
                db.commit()
                
            except Exception as memory_error:
                logging.error(f"Error syncing memory {vector_memory.get('id', 'unknown')}: {memory_error}")
                db.rollback()
                continue
        
        return {
            "message": "Memory synchronization completed",
            "total_vector_memories": len(vector_memories),
            "new_synced": synced_count,
            "updated": updated_count,
            "user_id": user_id
        }
        
    except Exception as e:
        logging.error(f"Memory sync failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/system")
async def get_system_memories(
    user_id: str,
    auto_sync: bool = True,
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    获取系统记忆列表（通过mem0核心创建的记忆）
    如果auto_sync=True，会自动从向量存储同步最新的记忆
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # 如果启用自动同步，先同步记忆
    if auto_sync:
        try:
            sync_result = await sync_memories_from_vector_store(user_id, False, db)
            logging.info(f"Auto-sync completed: {sync_result}")
        except Exception as e:
            logging.warning(f"Auto-sync failed: {e}")
    
    # 获取系统记忆（通过mem0_core app创建的记忆）
    system_app = db.query(App).filter(
        App.name == "mem0_core",
        App.owner_id == user.id
    ).first()
    
    if not system_app:
        return {
            "items": [],
            "total": 0,
            "page": page,
            "size": size,
            "pages": 0
        }
    
    # 构建查询
    query = db.query(Memory).filter(
        Memory.user_id == user.id,
        Memory.app_id == system_app.id,
        Memory.state == MemoryState.active
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).order_by(Memory.created_at.desc())
    
    # 分页
    total = query.count()
    offset = (page - 1) * size
    memories = query.offset(offset).limit(size).all()
    
    # 格式化响应
    formatted_memories = []
    for memory in memories:
        formatted_memory = {
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "state": memory.state.value,
            "app_name": "System (mem0 core)",
            "categories": [category.name for category in memory.categories],
            "metadata": memory.metadata_ or {},
            "is_system_memory": True
        }
        formatted_memories.append(formatted_memory)
    
    pages = (total + size - 1) // size
    
    return {
        "items": formatted_memories,
        "total": total,
        "page": page,
        "size": size,
        "pages": pages
    }


@router.get("/{memory_id}/related", response_model=Page[MemoryResponse])
async def get_related_memories(
    memory_id: UUID,
    user_id: str,
    params: Params = Depends(),
    db: Session = Depends(get_db)
):
    # Validate user
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the source memory
    memory = get_memory_or_404(db, memory_id)
    
    # Extract category IDs from the source memory
    category_ids = [category.id for category in memory.categories]
    
    if not category_ids:
        return Page.create([], total=0, params=params)
    
    # Build query for related memories
    query = db.query(Memory).distinct(Memory.id).filter(
        Memory.user_id == user.id,
        Memory.id != memory_id,
        Memory.state != MemoryState.deleted
    ).join(Memory.categories).filter(
        Category.id.in_(category_ids)
    ).options(
        joinedload(Memory.categories),
        joinedload(Memory.app)
    ).order_by(
        func.count(Category.id).desc(),
        Memory.created_at.desc()
    ).group_by(Memory.id)
    
    # ⚡ Force page size to be 5
    params = Params(page=params.page, size=5)
    
    return sqlalchemy_paginate(
        query,
        params,
        transformer=lambda items: [
            MemoryResponse(
                id=memory.id,
                content=memory.content,
                created_at=memory.created_at,
                state=memory.state.value,
                app_id=memory.app_id,
                app_name=memory.app.name if memory.app else None,
                categories=[category.name for category in memory.categories],
                metadata_=json.loads(memory.metadata_) if isinstance(memory.metadata_, str) and memory.metadata_ else (memory.metadata_ or {})
            )
            for memory in items
        ]
    )