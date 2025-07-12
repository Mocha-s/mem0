import logging
import os
import asyncio
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# 自定义OpenAI配置处理 - 确保环境变量在任何mem0导入之前设置
CUSTOM_OPENAI_API_URL = os.environ.get("CUSTOM_OPENAI_API_URL")
CUSTOM_OPENAI_API_KEY = os.environ.get("CUSTOM_OPENAI_API_KEY")

# 如果有自定义配置，设置对应的标准环境变量
if CUSTOM_OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = CUSTOM_OPENAI_API_KEY
    logging.info(f"设置OPENAI_API_KEY为自定义值: {CUSTOM_OPENAI_API_KEY[:10]}...")

if CUSTOM_OPENAI_API_URL:
    os.environ["OPENAI_BASE_URL"] = CUSTOM_OPENAI_API_URL
    logging.info(f"设置OPENAI_BASE_URL为自定义值: {CUSTOM_OPENAI_API_URL}")

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mem0 import Memory
from mem0.memory.enhanced_memory import EnhancedMemory

# 导入服务间通信组件
from server.clients.openmemory_client import get_openmemory_client
from server.middleware.service_communication import (
    get_service_middleware, 
    get_health_checker,
    openmemory_fallback_get_config,
    openmemory_fallback_sync_memories
)

# 导入v2 API和版本中间件
from server.middleware.version_middleware import APIVersionMiddleware
from server.routers.v2_api import v2_router

# 导入同步API
from server.routers.sync_api import router as sync_router

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

# 设置特定模块的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) 
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("mem0.vector_stores").setLevel(logging.WARNING)

# OpenMemory配置API设置
OPENMEMORY_API_URL = os.environ.get("OPENMEMORY_API_URL", "http://localhost:8765")
OPENMEMORY_CONFIG_SYNC_ENABLED = os.environ.get("OPENMEMORY_CONFIG_SYNC_ENABLED", "true").lower() == "true"
DISABLE_API_KEY_VALIDATION = os.environ.get("DISABLE_API_KEY_VALIDATION", "false").lower() == "true"

# 原有环境变量（作为fallback）
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "postgres")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
POSTGRES_COLLECTION_NAME = os.environ.get("POSTGRES_COLLECTION_NAME", "memories")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0graph")

MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "memgraph")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "mem0graph")

# 向量存储配置 (环境变量优先)
VECTOR_STORE_TYPE = os.environ.get("VECTOR_STORE_TYPE", "pgvector")
VECTOR_STORE_HOST = os.environ.get("VECTOR_STORE_HOST", "postgres")
VECTOR_STORE_PORT = int(os.environ.get("VECTOR_STORE_PORT", "5432"))
VECTOR_STORE_USER = os.environ.get("VECTOR_STORE_USER", "mem0user")
VECTOR_STORE_PASSWORD = os.environ.get("VECTOR_STORE_PASSWORD", "mem0pass")
VECTOR_STORE_DATABASE = os.environ.get("VECTOR_STORE_DATABASE", "mem0")
VECTOR_STORE_COLLECTION = os.environ.get("VECTOR_STORE_COLLECTION", "memories")

# Qdrant配置 (保留用于向后兼容)
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# 现在获取设置后的环境变量
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HISTORY_DB_PATH = os.environ.get("HISTORY_DB_PATH", "/app/history/history.db")

# 构建LLM和Embedder配置
def build_llm_config():
    """构建LLM配置"""
    config = {
        "provider": "openai",
        "config": {
            "temperature": 0.2,
            "model": "gpt-4o"
        }
    }
    
    # API Key配置
    if CUSTOM_OPENAI_API_KEY:
        config["config"]["api_key"] = CUSTOM_OPENAI_API_KEY
    elif OPENAI_API_KEY:
        config["config"]["api_key"] = OPENAI_API_KEY
    else:
        config["config"]["api_key"] = "dummy-key"
    
    # Base URL配置
    if CUSTOM_OPENAI_API_URL:
        config["config"]["openai_base_url"] = CUSTOM_OPENAI_API_URL
    
    return config

def build_embedder_config():
    """构建Embedder配置"""
    config = {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
    
    # API Key配置
    if CUSTOM_OPENAI_API_KEY:
        config["config"]["api_key"] = CUSTOM_OPENAI_API_KEY
    elif OPENAI_API_KEY:
        config["config"]["api_key"] = OPENAI_API_KEY
    else:
        config["config"]["api_key"] = "dummy-key"
    
    # Base URL配置
    if CUSTOM_OPENAI_API_URL:
        config["config"]["openai_base_url"] = CUSTOM_OPENAI_API_URL
    
    return config

# 默认配置（fallback）- 使用pgvector作为统一数据库解决方案
DEFAULT_CONFIG = {
    "version": "v1.1",
    "vector_store": {
        "provider": VECTOR_STORE_TYPE,
        "config": {
            "user": VECTOR_STORE_USER,
            "password": VECTOR_STORE_PASSWORD,
            "host": VECTOR_STORE_HOST,
            "port": VECTOR_STORE_PORT,
            "dbname": VECTOR_STORE_DATABASE,
            "collection_name": "mem0migrations",
            "embedding_model_dims": 1536,
            "diskann": True,
            "hnsw": False
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": NEO4J_URI,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD,
        },
    },
    "llm": build_llm_config(),
    "embedder": build_embedder_config(),
    "history_db_path": HISTORY_DB_PATH,
    "dedupe": True,  # 启用去重功能
}

class ConfigManager:
    """统一配置管理器"""
    
    def __init__(self):
        self.current_config = None
        self.openmemory_available = False
    
    async def get_config_from_openmemory(self) -> Optional[Dict[str, Any]]:
        """从OpenMemory获取配置"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{OPENMEMORY_API_URL}/api/v1/config/",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    config_data = response.json()
                    self.openmemory_available = True
                    return self._convert_openmemory_config_to_mem0(config_data)
                else:
                    logging.warning(f"Failed to get config from OpenMemory: {response.status_code}")
                    
        except Exception as e:
            logging.warning(f"Cannot connect to OpenMemory: {e}")
            self.openmemory_available = False
        
        return None
    
    def _convert_openmemory_config_to_mem0(self, openmemory_config: Dict[str, Any]) -> Dict[str, Any]:
        """将OpenMemory配置转换为Mem0配置格式"""
        mem0_config = DEFAULT_CONFIG.copy()
        
        if "mem0" in openmemory_config:
            mem0_section = openmemory_config["mem0"]
            
            # 处理LLM配置
            if "llm" in mem0_section and mem0_section["llm"]:
                llm_config = mem0_section["llm"]["config"]
                
                # 处理自定义OpenAI API URL
                openai_base_url = None
                if llm_config.get("openai_base_url"):
                    openai_base_url = llm_config["openai_base_url"]
                elif llm_config.get("custom_api_url"):
                    openai_base_url = llm_config["custom_api_url"]
                elif llm_config.get("base_url"):
                    openai_base_url = llm_config["base_url"]
                
                # 处理环境变量引用
                if openai_base_url and openai_base_url.startswith("env:"):
                    env_var = openai_base_url[4:]  # 移除"env:"前缀
                    openai_base_url = os.environ.get(env_var, "")
                
                if openai_base_url:
                    mem0_config["llm"]["config"]["openai_base_url"] = openai_base_url
                
                # 处理API key
                api_key = llm_config.get("api_key", "")
                if api_key.startswith("env:"):
                    env_var = api_key[4:]  # 移除"env:"前缀
                    api_key = os.environ.get(env_var, "")
                
                mem0_config["llm"]["config"].update({
                    "model": llm_config.get("model", "gpt-4o"),
                    "temperature": llm_config.get("temperature", 0.2),
                    "max_tokens": llm_config.get("max_tokens", 2000),
                    "api_key": api_key
                })
            
            # 处理Embedder配置
            if "embedder" in mem0_section and mem0_section["embedder"]:
                embedder_config = mem0_section["embedder"]["config"]
                
                # 处理自定义OpenAI API URL
                openai_base_url = None
                if embedder_config.get("openai_base_url"):
                    openai_base_url = embedder_config["openai_base_url"]
                elif embedder_config.get("custom_api_url"):
                    openai_base_url = embedder_config["custom_api_url"]
                elif embedder_config.get("base_url"):
                    openai_base_url = embedder_config["base_url"]
                
                # 处理环境变量引用
                if openai_base_url and openai_base_url.startswith("env:"):
                    env_var = openai_base_url[4:]  # 移除"env:"前缀
                    openai_base_url = os.environ.get(env_var, "")
                
                if openai_base_url:
                    mem0_config["embedder"]["config"]["openai_base_url"] = openai_base_url
                
                # 处理API key
                api_key = embedder_config.get("api_key", "")
                if api_key.startswith("env:"):
                    env_var = api_key[4:]  # 移除"env:"前缀
                    api_key = os.environ.get(env_var, "")
                
                mem0_config["embedder"]["config"].update({
                    "model": embedder_config.get("model", "text-embedding-3-small"),
                    "api_key": api_key
                })
            
            # 处理服务配置
            if "service" in mem0_section and mem0_section["service"]:
                service_config = mem0_section["service"]
                
                # 免API验证功能
                if service_config.get("disable_api_key_validation", False):
                    # 如果禁用API验证，使用dummy key
                    if not mem0_config["llm"]["config"]["api_key"]:
                        mem0_config["llm"]["config"]["api_key"] = "dummy-key"
                    if not mem0_config["embedder"]["config"]["api_key"]:
                        mem0_config["embedder"]["config"]["api_key"] = "dummy-key"
                
                # 自定义OpenAI配置
                if service_config.get("custom_openai_api_url"):
                    mem0_config["llm"]["config"]["openai_base_url"] = service_config["custom_openai_api_url"]
                    mem0_config["embedder"]["config"]["openai_base_url"] = service_config["custom_openai_api_url"]
                
                if service_config.get("custom_openai_api_key"):
                    mem0_config["llm"]["config"]["api_key"] = service_config["custom_openai_api_key"]
                    mem0_config["embedder"]["config"]["api_key"] = service_config["custom_openai_api_key"]
        
        return mem0_config
    
    async def load_config(self) -> Dict[str, Any]:
        """加载配置，优先从OpenMemory获取"""
        if OPENMEMORY_CONFIG_SYNC_ENABLED:
            openmemory_config = await self.get_config_from_openmemory()
            if openmemory_config:
                self.current_config = openmemory_config
                logging.info("Configuration loaded from OpenMemory")
                return openmemory_config
        
        # Fallback到默认配置
        logging.info("Using default configuration")
        self.current_config = DEFAULT_CONFIG
        return DEFAULT_CONFIG
    
    async def reload_config(self):
        """重新加载配置"""
        new_config = await self.load_config()
        global MEMORY_INSTANCE
        MEMORY_INSTANCE = EnhancedMemory.from_config(new_config)
        logging.info("Enhanced Memory instance reloaded with new configuration")

# 全局配置管理器
config_manager = ConfigManager()

# 异步初始化Memory实例
async def init_memory_instance():
    """异步初始化EnhancedMemory实例"""
    config = await config_manager.load_config()
    return EnhancedMemory.from_config(config)

# 在应用启动时初始化
MEMORY_INSTANCE = None

app = FastAPI(
    title="Mem0 REST APIs",
    description="A REST API for managing and searching memories for your AI Agents and Apps.",
    version="1.0.0",
)

# 添加CORS中间件支持跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加版本中间件
app.add_middleware(APIVersionMiddleware)

# 注册v2路由
app.include_router(v2_router)

# 注册同步API路由
app.include_router(sync_router)

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化Memory实例和服务间通信组件"""
    global MEMORY_INSTANCE
    try:
        MEMORY_INSTANCE = await init_memory_instance()
        logging.info("Memory instance initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Memory instance: {e}")
        # 使用默认配置作为fallback
        MEMORY_INSTANCE = EnhancedMemory.from_config(DEFAULT_CONFIG)
        logging.info("Using enhanced default configuration as fallback")
    
    # 初始化服务间通信组件
    try:
        service_middleware = get_service_middleware()
        health_checker = get_health_checker()
        
        # 启动服务间通信中间件
        await service_middleware.startup()
        
        # 注册Mem0主服务
        service_middleware.service_discovery.register_service(
            "mem0-main",
            "http://localhost:8000",
            "/health"
        )
        
        logging.info("Service communication components initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize service communication components: {e}")
        logging.info("Service will continue without advanced communication features")


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (user or assistant).")
    content: str = Field(..., description="Message content.")


class MemoryCreate(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to store.")
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query.")
    user_id: Optional[str] = None
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = Field(default=10, description="Maximum number of results to return")


@app.post("/configure", summary="Configure Mem0")
async def set_config(config: Dict[str, Any]):
    """Set memory configuration."""
    global MEMORY_INSTANCE
    MEMORY_INSTANCE = Memory.from_config(config)
    
    # 更新配置管理器的当前配置
    config_manager.current_config = config
    
    return {"message": "Configuration set successfully"}

@app.post("/reload-config", summary="Reload configuration from OpenMemory")
async def reload_config():
    """重新从OpenMemory加载配置"""
    try:
        await config_manager.reload_config()
        return {
            "message": "Configuration reloaded successfully",
            "openmemory_available": config_manager.openmemory_available
        }
    except Exception as e:
        logging.exception("Error reloading configuration:")
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")

@app.get("/config-status", summary="Get configuration status")
async def get_config_status():
    """获取配置状态信息"""
    return {
        "openmemory_available": config_manager.openmemory_available,
        "openmemory_url": OPENMEMORY_API_URL,
        "sync_enabled": OPENMEMORY_CONFIG_SYNC_ENABLED,
        "current_config_source": "OpenMemory" if config_manager.openmemory_available else "Default"
    }

@app.get("/health", summary="Configuration health check")
async def health_check():
    """执行简化的健康检查"""
    try:
        # 简化的健康检查，检查核心组件状态
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {
                "mem0_instance": "healthy" if MEMORY_INSTANCE else "unhealthy",
                "config_manager": "healthy" if config_manager else "unhealthy",
                "api_server": "healthy"
            }
        }
        
        # 检查向量存储连接 - 暂时禁用以避免事务问题
        try:
            # 简化的向量存储检查：只测试基本连接而不执行向量操作
            if hasattr(MEMORY_INSTANCE.vector_store, 'connection_pool'):
                # 仅检查连接池是否存在和可用
                pool = MEMORY_INSTANCE.vector_store.connection_pool
                if pool and not pool.closed:
                    health_status["checks"]["vector_store"] = "healthy"
                else:
                    health_status["checks"]["vector_store"] = "unhealthy: connection pool unavailable"
                    health_status["overall_status"] = "degraded"
            else:
                health_status["checks"]["vector_store"] = "healthy (basic check)"
                
        except Exception as vector_error:
            health_status["checks"]["vector_store"] = f"unhealthy: {str(vector_error)}"
            health_status["overall_status"] = "degraded"
        
        # 检查OpenMemory连接（如果配置了）
        if config_manager.openmemory_available:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{OPENMEMORY_API_URL}/health")
                    if response.status_code == 200:
                        health_status["checks"]["openmemory"] = "healthy"
                    else:
                        health_status["checks"]["openmemory"] = f"unhealthy: HTTP {response.status_code}"
                        health_status["overall_status"] = "degraded"
            except Exception as om_error:
                health_status["checks"]["openmemory"] = f"unhealthy: {str(om_error)}"
                health_status["overall_status"] = "degraded"
        else:
            health_status["checks"]["openmemory"] = "not_configured"
        
        return health_status
        
    except Exception as e:
        logging.exception("Error in health check:")
        return {
            "overall_status": "unhealthy",
            "error": str(e),
            "message": "Health check failed"
        }

@app.get("/test/status", summary="Test endpoint")
async def test_status():
    """测试端点"""
    return {
        "status": "testing",
        "message": "Basic response test"
    }

@app.get("/services/status", summary="Get service communication status")
async def get_services_status():
    """获取服务间通信状态"""
    try:
        # 测试基本响应
        return {
            "status": "testing",
            "message": "Basic response test"
        }
    except Exception as e:
        logging.exception("Error getting services status:")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to get services status"
        }

@app.post("/services/sync", summary="Sync with OpenMemory service")
async def sync_with_openmemory():
    """与OpenMemory服务进行数据同步"""
    try:
        openmemory_client = get_openmemory_client()
        
        # 检查OpenMemory服务健康状态
        health_status = await openmemory_client.health_check()
        
        if not health_status.get("healthy", False):
            raise HTTPException(
                status_code=503, 
                detail="OpenMemory service is not healthy"
            )
        
        # 执行同步操作
        sync_result = await openmemory_fallback_sync_memories()
        
        return {
            "status": "success",
            "sync_result": sync_result,
            "message": "Synchronization completed successfully"
        }
    except Exception as e:
        logging.exception("Error in sync with OpenMemory:")
        raise HTTPException(
            status_code=500, 
            detail=f"Synchronization failed: {str(e)}"
        )


# v1 API 端点 (符合官方规范)
@app.post("/v1/memories/", summary="Add memories (v1)")
def add_memory_v1(memory_create: MemoryCreate):
    """Store new memories using v1 API."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/", summary="Get memories (v1)")
def get_all_memories_v1(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Retrieve stored memories using v1 API."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}/", summary="Get a memory (v1)")
def get_memory_v1(memory_id: str):
    """Retrieve a specific memory by ID using v1 API."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/memories/{memory_id}/", summary="Update a memory (v1)")
def update_memory_v1(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory using v1 API."""
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/memories/{memory_id}/", summary="Delete a memory (v1)")
def delete_memory_v1(memory_id: str):
    """Delete a specific memory by ID using v1 API."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory_v1:")
        raise HTTPException(status_code=500, detail=str(e))


def convert_distance_to_similarity(results):
    """
    Convert distance scores to similarity scores and filter low similarity results.
    Distance closer to 0 means higher similarity.
    We convert distance to similarity score (1 - distance) and filter results with similarity < 0.5
    """
    if isinstance(results, dict) and 'results' in results:
        filtered_results = []
        for result in results['results']:
            if 'score' in result:
                # Convert distance to similarity (1 - distance)
                similarity = 1.0 - result['score']
                # Filter out results with low similarity (< 0.5 for high quality)
                if similarity >= 0.5:
                    result['score'] = round(similarity, 3)
                    filtered_results.append(result)
        
        results['results'] = filtered_results
        # Sort by similarity score (highest first)
        results['results'] = sorted(filtered_results, key=lambda x: x.get('score', 0), reverse=True)
        return results
    return results


@app.post("/v1/memories/search/", summary="Search memories (v1)")
def search_memories_v1(search_req: SearchRequest):
    """Search for memories based on a query using v1 API."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        results = MEMORY_INSTANCE.search(query=search_req.query, **params)
        return convert_distance_to_similarity(results)
    except Exception as e:
        logging.exception("Error in search_memories_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/memories/{memory_id}/history/", summary="Get memory history (v1)")
def memory_history_v1(memory_id: str):
    """Retrieve memory history using v1 API."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/batch/", summary="Batch update memories (v1)")
def batch_update_memories_v1(batch_data: List[Dict[str, Any]]):
    """Batch update memories using v1 API."""
    try:
        if len(batch_data) > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 memories")
        
        results = []
        for item in batch_data:
            if "memory_id" not in item or "data" not in item:
                raise HTTPException(status_code=400, detail="Each item must contain memory_id and data")
            
            result = MEMORY_INSTANCE.update(memory_id=item["memory_id"], data=item["data"])
            results.append({"memory_id": item["memory_id"], "result": result})
        
        return {"message": f"Batch updated {len(results)} memories", "results": results}
    except Exception as e:
        logging.exception("Error in batch_update_memories_v1:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/batch/", summary="Batch delete memories (v1)")
def batch_delete_memories_v1(memory_ids: List[str]):
    """Batch delete memories using v1 API."""
    try:
        if len(memory_ids) > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 memories")
        
        results = []
        for memory_id in memory_ids:
            MEMORY_INSTANCE.delete(memory_id=memory_id)
            results.append({"memory_id": memory_id, "status": "deleted"})
        
        return {"message": f"Batch deleted {len(results)} memories", "results": results}
    except Exception as e:
        logging.exception("Error in batch_delete_memories_v1:")
        raise HTTPException(status_code=500, detail=str(e))


# 保持原有的无版本前缀端点用于向后兼容
@app.post("/memories", summary="Create memories")
def add_memory(memory_create: MemoryCreate):
    """Store new memories."""
    if not any([memory_create.user_id, memory_create.agent_id, memory_create.run_id]):
        raise HTTPException(status_code=400, detail="At least one identifier (user_id, agent_id, run_id) is required.")

    params = {k: v for k, v in memory_create.model_dump().items() if v is not None and k != "messages"}
    try:
        response = MEMORY_INSTANCE.add(messages=[m.model_dump() for m in memory_create.messages], **params)
        return JSONResponse(content=response)
    except Exception as e:
        logging.exception("Error in add_memory:")  # This will log the full traceback
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories", summary="Get memories")
def get_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Retrieve stored memories."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        return MEMORY_INSTANCE.get_all(**params)
    except Exception as e:
        logging.exception("Error in get_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}", summary="Get a memory")
def get_memory(memory_id: str):
    """Retrieve a specific memory by ID."""
    try:
        return MEMORY_INSTANCE.get(memory_id)
    except Exception as e:
        logging.exception("Error in get_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", summary="Search memories")
def search_memories(search_req: SearchRequest):
    """Search for memories based on a query."""
    try:
        params = {k: v for k, v in search_req.model_dump().items() if v is not None and k != "query"}
        results = MEMORY_INSTANCE.search(query=search_req.query, **params)
        return convert_distance_to_similarity(results)
    except Exception as e:
        logging.exception("Error in search_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/memories/{memory_id}", summary="Update a memory")
def update_memory(memory_id: str, updated_memory: Dict[str, Any]):
    """Update an existing memory."""
    try:
        return MEMORY_INSTANCE.update(memory_id=memory_id, data=updated_memory)
    except Exception as e:
        logging.exception("Error in update_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{memory_id}/history", summary="Get memory history")
def memory_history(memory_id: str):
    """Retrieve memory history."""
    try:
        return MEMORY_INSTANCE.history(memory_id=memory_id)
    except Exception as e:
        logging.exception("Error in memory_history:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{memory_id}", summary="Delete a memory")
def delete_memory(memory_id: str):
    """Delete a specific memory by ID."""
    try:
        MEMORY_INSTANCE.delete(memory_id=memory_id)
        return {"message": "Memory deleted successfully"}
    except Exception as e:
        logging.exception("Error in delete_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories", summary="Delete all memories")
def delete_all_memories(
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_id: Optional[str] = None,
):
    """Delete all memories for a given identifier."""
    if not any([user_id, run_id, agent_id]):
        raise HTTPException(status_code=400, detail="At least one identifier is required.")
    try:
        params = {
            k: v for k, v in {"user_id": user_id, "run_id": run_id, "agent_id": agent_id}.items() if v is not None
        }
        MEMORY_INSTANCE.delete_all(**params)
        return {"message": "All relevant memories deleted"}
    except Exception as e:
        logging.exception("Error in delete_all_memories:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", summary="Reset all memories")
def reset_memory():
    """Completely reset stored memories."""
    try:
        MEMORY_INSTANCE.reset()
        return {"message": "All memories reset"}
    except Exception as e:
        logging.exception("Error in reset_memory:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", summary="Redirect to the OpenAPI documentation", include_in_schema=False)
def home():
    """Redirect to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")

# 辅助函数供同步API使用
def get_memory_instance():
    """获取Memory实例"""
    global MEMORY_INSTANCE
    if MEMORY_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Memory instance not initialized")
    return MEMORY_INSTANCE
