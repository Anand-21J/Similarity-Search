"""
Fashion AI Search - Main Application Entry Point
FastAPI application for fashion image similarity search using CLIP and FAISS
"""

# Import uvicorn for running the FastAPI application
import uvicorn

# Import FastAPI core class
from fastapi import FastAPI

# Import static file support (not used directly here but commonly needed)
from fastapi.staticfiles import StaticFiles

# Import Jinja2 templates for HTML rendering
from fastapi.templating import Jinja2Templates

# Import application configuration settings
from config import settings

# Import API router containing all route definitions
from api.routes import router as api_router

# Import startup handler for initializing models and data
from core.startup import startup_handler

# Import utility to ensure required directories exist
from utils.paths import ensure_directories

# Import ngrok manager for optional public tunneling
from utils.ngrok_manager import ngrok_manager

# ==============================
# Initialize FastAPI Application
# ==============================

# Create FastAPI app instance with metadata
app = FastAPI(
    title="Fashion AI Search",
    description="AI-powered fashion similarity search using CLIP embeddings",
    version="1.0.0"
)

# ==============================
# Setup Templates
# ==============================

# Initialize Jinja2 templates directory
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# ==============================
# Ensure Required Directories
# ==============================

# Create necessary directories if they do not already exist
ensure_directories()

# ==============================
# Register Routes
# ==============================

# Register API routes with the FastAPI application
app.include_router(api_router)

# ==============================
# Startup Event Handler
# ==============================

@app.on_event("startup")
async def startup_event():
    """Initialize models and data on application startup"""
    # Call the startup handler to load models, dataset, and FAISS index
    await startup_handler()

# ==============================
# Main Entry Point
# ==============================

if __name__ == "__main__":
    # Read port from settings
    PORT = settings.PORT
    
    # Start ngrok tunnel if enabled in configuration
    if settings.USE_NGROK:
        public_url = ngrok_manager.start_tunnel(
            port=PORT,
            auth_token=settings.NGROK_AUTH_TOKEN
        )
    
    try:
        # Log server startup information
        print(f"🔥 Starting FastAPI server on {settings.HOST}:{PORT}...")
        print(f"📍 Local URL: http://localhost:{PORT}")
        print("-" * 60 + "\n")
        
        # Run the FastAPI application using uvicorn
        uvicorn.run(
            app,
            host=settings.HOST,
            port=PORT,
            log_level="info"
        )
    except KeyboardInterrupt:
        # Handle graceful shutdown on user interrupt
        print("\n\n🛑 Server stopped by user")
    finally:
        # Stop ngrok tunnel if it was started
        if settings.USE_NGROK:
            ngrok_manager.stop_tunnel()
