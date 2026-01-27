"""
Startup Handler
Manages application initialization on startup
"""

# Import the global model manager responsible for loading the CLIP model
from core.model_manager import model_manager

# Import the global dataset manager responsible for dataset, embeddings, and FAISS index
from core.dataset_manager import dataset_manager


async def startup_handler() -> None:
    """
    Initialize all required components on application startup
    
    Loads:
        1. CLIP model and preprocessing
        2. Fashion dataset (or creates embeddings if needed)
        3. FAISS index for similarity search
    """
    # Print startup banner for better visibility in logs
    print("\n" + "="*60)
    print("🚀 Initializing Fashion AI Search Application")
    print("="*60 + "\n")
    
    # Step 1: Load the CLIP model
    print("📦 Step 1: Loading CLIP Model")
    print("-" * 60)
    
    # Load CLIP model and preprocessing transforms
    model_manager.load_model()
    print()
    
    # Step 2: Load dataset, embeddings, and FAISS index (or create if not available)
    print("📦 Step 2: Loading Dataset and Embeddings")
    print("-" * 60)
    
    # Load existing artifacts or create new ones using the loaded model
    dataset_manager.load_or_create(model_manager)
    print()
    
    # Print successful initialization message
    print("="*60)
    print("✅ Application initialized successfully!")
    print("="*60 + "\n")
