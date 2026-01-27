"""
Path Utility Functions
Helper functions for directory and file management
"""

# Import application settings containing directory paths
from config import settings


def ensure_directories() -> None:
    """
    Create required directories if they don't exist
    
    Creates:
        - DATA_DIR: For storing dataset metadata
        - ARTIFACTS_DIR: For storing embeddings and FAISS index
        - TEMPLATES_DIR: For HTML templates (usually pre-existing)
    """
    # Ensure the data directory exists (creates it if missing)
    settings.DATA_DIR.mkdir(exist_ok=True)
    
    # Ensure the artifacts directory exists (creates it if missing)
    settings.ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Ensure the templates directory exists (creates it if missing)
    settings.TEMPLATES_DIR.mkdir(exist_ok=True)
    
    # Log directory verification
    print("📁 Required directories verified")
