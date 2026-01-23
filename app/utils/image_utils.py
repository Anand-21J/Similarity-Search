"""
Image Utility Functions
Helper functions for image processing and color analysis
"""

import io
import base64
import numpy as np
from PIL import Image

def get_avg_color(image: Image.Image) -> np.ndarray:
    """
    Calculate the average RGB color of an image
    
    Args:
        image: PIL Image object
        
    Returns:
        NumPy array of shape (3,) containing average RGB values
    """
    img_np = np.array(image)
    return img_np.mean(axis=(0, 1))

def color_similarity(color1: np.ndarray, color2: np.ndarray) -> float:
    """
    Calculate color similarity between two RGB colors
    Uses Euclidean distance normalized to 0-1 scale
    
    Args:
        color1: First RGB color as NumPy array
        color2: Second RGB color as NumPy array
        
    Returns:
        Similarity score between 0 and 1 (1 = identical colors)
    """
    # Calculate Euclidean distance
    distance = np.linalg.norm(color1 - color2)
    
    # Normalize to 0-1 scale (max distance in RGB space is ~441)
    # Using 255 * sqrt(3) ≈ 441 as max possible distance
    similarity = max(0, 1 - distance / 255)
    
    return similarity

def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64-encoded data URL
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64-encoded image as data URL string
    """
    # Create in-memory buffer
    buffered = io.BytesIO()
    
    # Save image to buffer as PNG
    image.save(buffered, format="PNG")
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Return as data URL
    return f"data:image/png;base64,{img_str}"