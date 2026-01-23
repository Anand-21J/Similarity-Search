"""
Path Utility Functions
Helper functions for directory and file management
"""

from config import settings

def ensure_directories() -> None:
    """
    Create required directories if they don't exist
    
    Creates:
        - DATA_DIR: For storing dataset metadata
        - ARTIFACTS_DIR: For storing embeddings and FAISS index
        - TEMPLATES_DIR: For HTML templates (usually pre-existing)
    """
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.ARTIFACTS_DIR.mkdir(exist_ok=True)
    settings.TEMPLATES_DIR.mkdir(exist_ok=True)
    
    print("📁 Required directories verified")