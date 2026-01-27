"""
API Routes for Fashion AI Search
Handles all HTTP endpoints for the application
"""

# Import FastAPI utilities for routing, file uploads, forms, exceptions, and request handling
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request

# Import response classes for returning HTML and JSON responses
from fastapi.responses import HTMLResponse, JSONResponse

# Import Jinja2 template support for rendering HTML pages
from fastapi.templating import Jinja2Templates

# Import application settings (paths, model names, dataset names, etc.)
from config import settings

# Import the search service that performs image similarity search
from services.search_service import SearchService

# Import the global model manager (handles CLIP model, device, etc.)
from core.model_manager import model_manager

# Import the global dataset manager (handles embeddings, FAISS index, dataset)
from core.dataset_manager import dataset_manager

# ==============================
# Initialize Router and Templates
# ==============================

# Create an APIRouter instance to group related routes
router = APIRouter()

# Initialize Jinja2 templates using the templates directory from settings
templates = Jinja2Templates(directory=str(settings.TEMPLATES_DIR))

# ==============================
# Web Interface Routes
# ==============================

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serve the main HTML page with system statistics
    
    Args:
        request: FastAPI request object
        
    Returns:
        TemplateResponse with index.html
    """
    # Render the index.html template and pass system stats as context
    return templates.TemplateResponse("index.html", {
        "request": request,  # Required by Jinja2Templates for rendering
        "sample_size": len(dataset_manager.sampled_dataset) if dataset_manager.sampled_dataset else 0,  # Number of images loaded
        "embedding_dim": dataset_manager.image_embeddings.shape[1] if dataset_manager.image_embeddings is not None else 0,  # Embedding vector size
        "index_size": dataset_manager.index.ntotal if dataset_manager.index else 0,  # Number of vectors in FAISS index
        "device": model_manager.device.upper()  # Device being used (CPU / CUDA)
    })

# ==============================
# API Routes
# ==============================

@router.post("/api/search")
async def search_similar(
    file: UploadFile = File(..., description="Fashion image to search for"),  # Uploaded image file
    top_k: int = Form(5, description="Number of similar items to return")     # Number of results to retrieve
):
    """
    Search for similar fashion items based on uploaded image
    
    Args:
        file: Uploaded image file
        top_k: Number of results to return
        
    Returns:
        JSONResponse with search results
        
    Raises:
        HTTPException: If search fails
    """
    try:
        # Read the uploaded image file into bytes
        contents = await file.read()
        
        # Initialize the search service with model and dataset managers
        search_service = SearchService(
            model_manager=model_manager,
            dataset_manager=dataset_manager
        )
        
        # Perform similarity search using the uploaded image
        results = await search_service.search(
            image_bytes=contents,
            top_k=top_k
        )
        
        # Return search results as JSON
        return JSONResponse(content=results)
    
    except Exception as e:
        # Catch any error during search and return HTTP 500 response
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/api/stats")
async def get_stats():
    """
    Get system statistics and configuration
    
    Returns:
        JSONResponse with system stats
    """
    # Return current system stats and configuration details
    return JSONResponse(content={
        "sample_size": len(dataset_manager.sampled_dataset) if dataset_manager.sampled_dataset else 0,  # Dataset size
        "embedding_dim": dataset_manager.image_embeddings.shape[1] if dataset_manager.image_embeddings is not None else 0,  # Embedding dimension
        "index_size": dataset_manager.index.ntotal if dataset_manager.index else 0,  # FAISS index size
        "device": model_manager.device,  # Device used by the model
        "model_name": settings.CLIP_MODEL_NAME,  # CLIP model name from config
        "dataset_name": settings.DATASET_NAME    # Dataset name from config
    })