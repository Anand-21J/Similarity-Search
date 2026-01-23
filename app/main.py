"""
Fashion AI Search - Main Application Entry Point
FastAPI application for fashion image similarity search using CLIP and FAISS
"""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import settings
from api.routes import router as api_router
from core.startup import startup_handler
from utils.paths import ensure_directories

# ==============================
# Initialize FastAPI Application
# ==============================

app = FastAPI(
    title="Fashion AI Search",
    description="AI-powered fashion similarity search using CLIP embeddings",
    version="1.0.0"
)

# ==============================
# Setup Templates
# ==============================

templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# ==============================
# Ensure Required Directories
# ==============================

ensure_directories()

# ==============================
# Register Routes
# ==============================

app.include_router(api_router)

# ==============================
# Startup Event Handler
# ==============================

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on application startup"""
    await startup_handler()

# ==============================
# Main Entry Point
# ==============================

if __name__ == "__main__":
    # Configuration
    PORT = settings.PORT
    
    # Optional: ngrok tunnel for cloud environments
    if settings.USE_NGROK:
        try:
            from pyngrok import ngrok
            import nest_asyncio
            
            nest_asyncio.apply()
            
            print("🚀 Starting ngrok tunnel...")
            public_url = ngrok.connect(PORT)
            print(f"✅ Public URL: {public_url}")
            print(f"🌐 Ngrok Dashboard: http://127.0.0.1:4040")
            print(f"📱 Share this link: {public_url}")
            print("-" * 60)
        except ImportError:
            print("⚠️ pyngrok not installed. Running without tunnel.")
    
    # Start FastAPI server
    print(f"🔥 Starting FastAPI server on port {PORT}...")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=PORT,
        log_level="info"
    )