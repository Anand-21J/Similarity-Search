"""
Model Manager for CLIP Model
Handles loading and managing the CLIP model for image embeddings
"""

import torch
import open_clip
from typing import Optional

from config import settings

class ModelManager:
    """Manages CLIP model loading and inference"""
    
    def __init__(self):
        """Initialize ModelManager with empty state"""
        self.model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[callable] = None
        self.device: str = self._get_device()
    
    def _get_device(self) -> str:
        """
        Determine the best available device for computation
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if settings.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return settings.DEVICE
    
    def load_model(self) -> None:
        """
        Load CLIP model and preprocessing transforms
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading CLIP model '{settings.CLIP_MODEL_NAME}' on {self.device}...")
            
            # Load model and preprocessing
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=settings.CLIP_MODEL_NAME,
                pretrained=settings.CLIP_PRETRAINED
            )
            
            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("✅ CLIP model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized embedding for an image tensor
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Normalized embedding tensor
        """
        with torch.no_grad():
            # Generate embedding
            embedding = self.model.encode_image(image_tensor)
            
            # Normalize embedding for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        return embedding
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.preprocess is not None

# ==============================
# Global Model Manager Instance
# ==============================

model_manager = ModelManager()