"""
Dataset Manager for Fashion Images
Handles dataset loading, embedding generation, and FAISS index management
"""

import json
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional

from config import settings
from core.model_manager import ModelManager

class DatasetManager:
    """Manages fashion dataset, embeddings, and FAISS index"""
    
    def __init__(self):
        """Initialize DatasetManager with empty state"""
        self.sampled_dataset: Optional[object] = None
        self.image_embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
    
    def load_or_create(self, model_manager: ModelManager) -> None:
        """
        Load existing embeddings and index, or create new ones
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        # Check if all required files exist
        if self._artifacts_exist():
            print("📦 Found existing embeddings and index. Loading...")
            self._load_existing()
        else:
            print("🔨 Artifacts not found. Creating new ones...")
            self._create_new(model_manager)
    
    def _artifacts_exist(self) -> bool:
        """
        Check if all required artifact files exist
        
        Returns:
            True if all files exist, False otherwise
        """
        return (
            settings.FAISS_INDEX_PATH.exists() and
            settings.EMBEDDINGS_PATH.exists() and
            settings.METADATA_PATH.exists()
        )
    
    def _load_existing(self) -> None:
        """Load existing dataset, embeddings, and FAISS index"""
        # Load metadata
        with open(settings.METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(
            settings.DATASET_NAME,
            split="train"
        )
        self.sampled_dataset = dataset.select(range(metadata['sample_size']))
        
        # Load embeddings
        self.image_embeddings = np.load(settings.EMBEDDINGS_PATH)
        
        # Load FAISS index
        self.index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
        
        print(f"✅ Loaded {len(self.sampled_dataset)} samples with {self.index.ntotal} vectors")
    
    def _create_new(self, model_manager: ModelManager) -> None:
        """
        Create new embeddings and FAISS index
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        # Load full dataset
        print("📥 Loading dataset from Hugging Face...")
        dataset = load_dataset(
            settings.DATASET_NAME,
            split="train"
        )
        
        print(f"Total images in dataset: {len(dataset)}")
        
        # Sample dataset
        self.sampled_dataset = dataset.select(range(settings.SAMPLE_SIZE))
        print(f"Sampled dataset size: {len(self.sampled_dataset)}")
        
        # Save metadata
        self._save_metadata()
        
        # Generate embeddings
        print("🧮 Generating embeddings...")
        self._generate_embeddings(model_manager)
        
        # Build FAISS index
        print("🔍 Building FAISS index...")
        self._build_faiss_index()
    
    def _save_metadata(self) -> None:
        """Save dataset sampling metadata"""
        metadata = {
            "dataset_name": settings.DATASET_NAME,
            "sample_size": settings.SAMPLE_SIZE,
            "indices": list(range(settings.SAMPLE_SIZE))
        }
        
        with open(settings.METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print("💾 Metadata saved")
    
    def _generate_embeddings(self, model_manager: ModelManager) -> None:
        """
        Generate CLIP embeddings for all images
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        embeddings_list = []
        
        for sample in tqdm(self.sampled_dataset, desc="Generating embeddings"):
            image = sample["image"]
            
            # Preprocess image
            image_tensor = model_manager.preprocess(image).unsqueeze(0).to(model_manager.device)
            
            # Generate embedding
            embedding = model_manager.encode_image(image_tensor)
            
            # Store embedding
            embeddings_list.append(embedding.cpu().numpy())
        
        # Convert to numpy array
        self.image_embeddings = np.vstack(embeddings_list)
        
        # Save embeddings
        np.save(settings.EMBEDDINGS_PATH, self.image_embeddings)
        print(f"💾 Embeddings saved. Shape: {self.image_embeddings.shape}")
    
    def _build_faiss_index(self) -> None:
        """Build and save FAISS index for similarity search"""
        # Get embedding dimension
        embedding_dim = self.image_embeddings.shape[1]
        
        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add embeddings to index
        self.index.add(self.image_embeddings)
        
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        
        # Save index
        faiss.write_index(self.index, str(settings.FAISS_INDEX_PATH))
        print(f"💾 FAISS index saved to {settings.FAISS_INDEX_PATH}")
    
    def search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """
        Search for k nearest neighbors using FAISS
        
        Args:
            query_embedding: Query embedding vector
            k: Number of neighbors to return
            
        Returns:
            Tuple of (scores, indices)
        """
        return self.index.search(query_embedding, k)

# ==============================
# Global Dataset Manager Instance
# ==============================

dataset_manager = DatasetManager()