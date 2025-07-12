import datetime
import logging
from uuid import uuid4
from fastapi import FastAPI
from app.database import engine, Base, SessionLocal
from app.mcp_server import setup_mcp_server
from app.routers import memories_router, apps_router, stats_router, config_router, memory_sync_router
from app.routers.mcp_clients import router as mcp_clients_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
from app.config import DEFAULT_APP_ID, USER_ID
from app.models import App, User

# 配置日志级别以减少冗余输出
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.ERROR)

app = FastAPI(title="OpenMemory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create all tables
Base.metadata.create_all(bind=engine)

# 添加根级别健康检查端点
@app.get("/health")
async def health_check():
    """OpenMemory API健康检查"""
    return {
        "status": "healthy",
        "service": "openmemory-api",
        "version": "1.0.0"
    }

# Check for USER_ID and create default user if needed
def create_default_user():
    db = SessionLocal()
    try:
        # Check if user exists
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            # Create default user
            user = User(
                id=uuid4(),
                user_id=USER_ID,
                name="Default User",
                created_at=datetime.datetime.now(datetime.UTC)
            )
            db.add(user)
            db.commit()
    finally:
        db.close()

# Check for DEFAULT_APP_ID and create default app if needed
def create_default_app():
    db = SessionLocal()
    try:
        # Get the default user first
        user = db.query(User).filter(User.user_id == USER_ID).first()
        if not user:
            return  # User doesn't exist, skip app creation
        
        # Check if default app exists
        app = db.query(App).filter(App.id == DEFAULT_APP_ID).first()
        if not app:
            # Create default app
            app = App(
                id=DEFAULT_APP_ID,
                name="Default App",
                owner_id=user.id,
                is_active=True,
                created_at=datetime.datetime.now(datetime.UTC)
            )
            db.add(app)
            db.commit()
    finally:
        db.close()

# Create default user on startup
create_default_user()
create_default_app()

# Setup MCP server
setup_mcp_server(app)

# Include routers - order matters! More specific routes first
app.include_router(memory_sync_router)  # Must be before memories_router to avoid conflicts
app.include_router(memories_router)
app.include_router(apps_router)
app.include_router(stats_router)
app.include_router(config_router)
app.include_router(mcp_clients_router)

# Add pagination support
add_pagination(app)
