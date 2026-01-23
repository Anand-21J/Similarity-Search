"""
Startup Handler
Manages application initialization on startup
"""

from core.model_manager import model_manager
from core.dataset_manager import dataset_manager

async def startup_handler() -> None:
    """
    Initialize all required components on application startup
    
    Loads:
        1. CLIP model and preprocessing
        2. Fashion dataset (or creates embeddings if needed)
        3. FAISS index for similarity search
    """
    print("\n" + "="*60)
    print("🚀 Initializing Fashion AI Search Application")
    print("="*60 + "\n")
    
    # Load CLIP model
    print("📦 Step 1: Loading CLIP Model")
    print("-" * 60)
    model_manager.load_model()
    print()
    
    # Load or create dataset and embeddings
    print("📦 Step 2: Loading Dataset and Embeddings")
    print("-" * 60)
    dataset_manager.load_or_create(model_manager)
    print()
    
    print("="*60)
    print("✅ Application initialized successfully!")
    print("="*60 + "\n")