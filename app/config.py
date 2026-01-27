"""
Configuration Settings for Fashion AI Search
Centralized configuration management for the application
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application configuration settings"""
    
    # ==============================
    # Model Configuration
    # ==============================
    
    # CLIP model settings
    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "openai"
    
    # Dataset settings
    DATASET_NAME: str = "ashraq/fashion-product-images-small"
    SAMPLE_SIZE: int = 1000
    
    # Search parameters
    TOP_N: int = 30  # Initial candidates to fetch
    SIMILARITY_THRESHOLD: float = 0.65  # Minimum CLIP score
    
    # Score weights
    CLIP_WEIGHT: float = 0.7
    COLOR_WEIGHT: float = 0.3
    
    # ==============================
    # Directory Configuration
    # ==============================
    
    # Base directories
    DATA_DIR: Path = Path("data")
    ARTIFACTS_DIR: Path = Path("artifacts")
    TEMPLATES_DIR: Path = Path("templates")
    
    # File paths
    @property
    def FAISS_INDEX_PATH(self) -> Path:
        return self.ARTIFACTS_DIR / "fashion_faiss.index"
    
    @property
    def EMBEDDINGS_PATH(self) -> Path:
        return self.ARTIFACTS_DIR / "image_embeddings.npy"
    
    @property
    def METADATA_PATH(self) -> Path:
        return self.DATA_DIR / "sampled_dataset_info.json"
    
    # ==============================
    # Server Configuration
    # ==============================
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Enable / disable ngrok
    USE_NGROK: bool = False
    
    # OPTIONAL ngrok auth token
    # If None, ngrok will try to run without authentication
    NGROK_AUTH_TOKEN: Optional[str] = None
    
    # ==============================
    # Device Configuration
    # ==============================
    
    # Will be set at runtime based on availability
    DEVICE: str = "auto"  # 'auto', 'cuda', or 'cpu'
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# ==============================
# Global Settings Instance
# ==============================

settings = Settings()
