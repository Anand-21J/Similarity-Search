"""
Create these empty __init__.py files in each directory:

1. api/__init__.py
2. core/__init__.py
3. services/__init__.py
4. utils/__init__.py

These files make Python treat the directories as packages.
They can be empty or contain package-level imports.
"""

# api/__init__.py
"""API package for Fashion AI Search"""

# core/__init__.py
"""Core components for Fashion AI Search"""

# services/__init__.py
"""Services for Fashion AI Search"""

# utils/__init__.py
"""Utility functions for Fashion AI Search"""
from app.utils.ngrok_manager import ngrok_manager
from app.utils.image_utils import get_avg_color, color_similarity, image_to_base64
from app.utils.paths import ensure_directories

__all__ = [
    'ngrok_manager',
    'get_avg_color',
    'color_similarity', 
    'image_to_base64',
    'ensure_directories'
]