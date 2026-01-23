"""
API Routes for Fashion AI Search
Handles all HTTP endpoints for the application
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from config import settings
from services.search_service import SearchService
from core.model_manager import model_manager
from core.dataset_manager import dataset_manager

# ==============================
# Initialize Router and Templates
# ==============================

router = APIRouter()
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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_size": len(dataset_manager.sampled_dataset) if dataset_manager.sampled_dataset else 0,
        "embedding_dim": dataset_manager.image_embeddings.shape[1] if dataset_manager.image_embeddings is not None else 0,
        "index_size": dataset_manager.index.ntotal if dataset_manager.index else 0,
        "device": model_manager.device.upper()
    })

# ==============================
# API Routes
# ==============================

@router.post("/api/search")
async def search_similar(
    file: UploadFile = File(..., description="Fashion image to search for"),
    top_k: int = Form(5, description="Number of similar items to return")
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
        # Read uploaded file
        contents = await file.read()
        
        # Perform search using SearchService
        search_service = SearchService(
            model_manager=model_manager,
            dataset_manager=dataset_manager
        )
        
        results = await search_service.search(
            image_bytes=contents,
            top_k=top_k
        )
        
        return JSONResponse(content=results)
    
    except Exception as e:
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
    return JSONResponse(content={
        "sample_size": len(dataset_manager.sampled_dataset) if dataset_manager.sampled_dataset else 0,
        "embedding_dim": dataset_manager.image_embeddings.shape[1] if dataset_manager.image_embeddings is not None else 0,
        "index_size": dataset_manager.index.ntotal if dataset_manager.index else 0,
        "device": model_manager.device,
        "model_name": settings.CLIP_MODEL_NAME,
        "dataset_name": settings.DATASET_NAME
    })