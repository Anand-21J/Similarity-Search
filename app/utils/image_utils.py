"""
Image Utility Functions
Helper functions for image processing and color analysis
"""

# Import I/O utilities for in-memory byte streams
import io

# Import base64 for encoding images into string format
import base64

# Import NumPy for numerical operations
import numpy as np

# Import PIL Image for image handling
from PIL import Image


def get_avg_color(image: Image.Image) -> np.ndarray:
    """
    Calculate the average RGB color of an image
    
    Args:
        image: PIL Image object
        
    Returns:
        NumPy array of shape (3,) containing average RGB values
    """
    # Convert PIL Image to NumPy array
    img_np = np.array(image)
    
    # Compute mean across height and width dimensions (average RGB)
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
    # Calculate Euclidean distance between the two RGB vectors
    distance = np.linalg.norm(color1 - color2)
    
    # Normalize distance to a similarity score between 0 and 1
    # Maximum possible RGB distance is approximately 255 * sqrt(3)
    similarity = max(0, 1 - distance / 255)
    
    # Return similarity score
    return similarity


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64-encoded data URL
    
    Args:
        image: PIL Image object
        
    Returns:
        Base64-encoded image as data URL string
    """
    # Create an in-memory bytes buffer
    buffered = io.BytesIO()
    
    # Save the image into the buffer in PNG format
    image.save(buffered, format="PNG")
    
    # Encode the image bytes to a base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Return the base64 string formatted as a data URL
    return f"data:image/png;base64,{img_str}"
