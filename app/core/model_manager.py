"""
Model Manager for CLIP Model
Handles loading and managing the CLIP model for image embeddings
"""

# Import PyTorch for tensor operations and model handling
import torch

# Import OpenCLIP for loading CLIP models and preprocessing transforms
import open_clip

# Import typing utilities
from typing import Optional

# Import application settings (device, model name, pretrained weights, etc.)
from config import settings


class ModelManager:
    """Manages CLIP model loading and inference"""
    
    def __init__(self):
        """Initialize ModelManager with empty state"""
        # Holds the loaded CLIP model
        self.model: Optional[torch.nn.Module] = None
        
        # Holds the image preprocessing function
        self.preprocess: Optional[callable] = None
        
        # Determine which device (CPU / CUDA) to use
        self.device: str = self._get_device()
    
    def _get_device(self) -> str:
        """
        Determine the best available device for computation
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        # Automatically choose device if set to "auto" in settings
        if settings.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        # Otherwise, use the device specified in settings
        return settings.DEVICE
    
    def load_model(self) -> None:
        """
        Load CLIP model and preprocessing transforms
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Log model loading details
            print(f"Loading CLIP model '{settings.CLIP_MODEL_NAME}' on {self.device}...")
            
            # Create CLIP model and preprocessing transforms
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=settings.CLIP_MODEL_NAME,
                pretrained=settings.CLIP_PRETRAINED
            )
            
            # Move model to the selected device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            print("✅ CLIP model loaded successfully")
            
        except Exception as e:
            # Raise a clear error if model loading fails
            raise RuntimeError(f"Failed to load CLIP model: {str(e)}")
    
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate normalized embedding for an image tensor
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Normalized embedding tensor
        """
        # Disable gradient computation for inference
        with torch.no_grad():
            # Generate image embedding using CLIP
            embedding = self.model.encode_image(image_tensor)
            
            # Normalize embedding to unit length for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
        # Return the normalized embedding
        return embedding
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        # Model is considered loaded if both model and preprocess are available
        return self.model is not None and self.preprocess is not None


# ==============================
# Global Model Manager Instance
# ==============================

# Create a global ModelManager instance for reuse across the application
model_manager = ModelManager()
