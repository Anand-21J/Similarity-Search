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
from utils.ngrok_manager import ngrok_manager

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
    
    # Start ngrok tunnel if enabled
    if settings.USE_NGROK:
        public_url = ngrok_manager.start_tunnel(
            port=PORT,
            auth_token=settings.NGROK_AUTH_TOKEN
        )
    
    try:
        # Start FastAPI server
        print(f"🔥 Starting FastAPI server on {settings.HOST}:{PORT}...")
        print(f"📍 Local URL: http://localhost:{PORT}")
        print("-" * 60 + "\n")
        
        uvicorn.run(
            app,
            host=settings.HOST,
            port=PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    finally:
        # Clean up ngrok tunnel
        if settings.USE_NGROK:
            ngrok_manager.stop_tunnel()