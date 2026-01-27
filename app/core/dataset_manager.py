"""
Dataset Manager for Fashion Images
Handles dataset loading, embedding generation, and FAISS index management
"""

# Import standard library modules
import json

# Import numerical computing library
import numpy as np

# Import FAISS for similarity search indexing
import faiss

# Import Hugging Face datasets loader
from datasets import load_dataset

# Import progress bar utility
from tqdm import tqdm

# Import typing utilities
from typing import Optional

# Import application settings (paths, dataset name, sample size, etc.)
from config import settings

# Import ModelManager for image preprocessing and embedding generation
from core.model_manager import ModelManager


class DatasetManager:
    """Manages fashion dataset, embeddings, and FAISS index"""
    
    def __init__(self):
        """Initialize DatasetManager with empty state"""
        # Stores the sampled subset of the dataset
        self.sampled_dataset: Optional[object] = None
        
        # Stores image embeddings as a NumPy array
        self.image_embeddings: Optional[np.ndarray] = None
        
        # Stores the FAISS index for similarity search
        self.index: Optional[faiss.Index] = None
    
    def load_or_create(self, model_manager: ModelManager) -> None:
        """
        Load existing embeddings and index, or create new ones
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        # Check whether embeddings, index, and metadata already exist
        if self._artifacts_exist():
            print("📦 Found existing embeddings and index. Loading...")
            # Load previously saved artifacts
            self._load_existing()
        else:
            print("🔨 Artifacts not found. Creating new ones...")
            # Create embeddings and index from scratch
            self._create_new(model_manager)
    
    def _artifacts_exist(self) -> bool:
        """
        Check if all required artifact files exist
        
        Returns:
            True if all files exist, False otherwise
        """
        # Verify existence of FAISS index, embeddings, and metadata files
        return (
            settings.FAISS_INDEX_PATH.exists() and
            settings.EMBEDDINGS_PATH.exists() and
            settings.METADATA_PATH.exists()
        )
    
    def _load_existing(self) -> None:
        """Load existing dataset, embeddings, and FAISS index"""
        # Load metadata JSON file
        with open(settings.METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        
        # Load full dataset from Hugging Face
        print("Loading dataset...")
        dataset = load_dataset(
            settings.DATASET_NAME,
            split="train"
        )
        
        # Select the same subset of data used during embedding creation
        self.sampled_dataset = dataset.select(range(metadata['sample_size']))
        
        # Load saved image embeddings from disk
        self.image_embeddings = np.load(settings.EMBEDDINGS_PATH)
        
        # Load FAISS index from disk
        self.index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
        
        # Log successful loading
        print(f"✅ Loaded {len(self.sampled_dataset)} samples with {self.index.ntotal} vectors")
    
    def _create_new(self, model_manager: ModelManager) -> None:
        """
        Create new embeddings and FAISS index
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        # Load the full dataset from Hugging Face
        print("📥 Loading dataset from Hugging Face...")
        dataset = load_dataset(
            settings.DATASET_NAME,
            split="train"
        )
        
        # Print total dataset size
        print(f"Total images in dataset: {len(dataset)}")
        
        # Sample a subset of the dataset based on configuration
        self.sampled_dataset = dataset.select(range(settings.SAMPLE_SIZE))
        print(f"Sampled dataset size: {len(self.sampled_dataset)}")
        
        # Save dataset sampling metadata to disk
        self._save_metadata()
        
        # Generate image embeddings using the model
        print("🧮 Generating embeddings...")
        self._generate_embeddings(model_manager)
        
        # Build FAISS index from generated embeddings
        print("🔍 Building FAISS index...")
        self._build_faiss_index()
    
    def _save_metadata(self) -> None:
        """Save dataset sampling metadata"""
        # Create metadata dictionary
        metadata = {
            "dataset_name": settings.DATASET_NAME,
            "sample_size": settings.SAMPLE_SIZE,
            "indices": list(range(settings.SAMPLE_SIZE))
        }
        
        # Write metadata to JSON file
        with open(settings.METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Confirm metadata save
        print("💾 Metadata saved")
    
    def _generate_embeddings(self, model_manager: ModelManager) -> None:
        """
        Generate CLIP embeddings for all images
        
        Args:
            model_manager: Initialized ModelManager instance
        """
        # List to store embeddings for each image
        embeddings_list = []
        
        # Iterate over each sample in the dataset with a progress bar
        for sample in tqdm(self.sampled_dataset, desc="Generating embeddings"):
            # Extract image from dataset sample
            image = sample["image"]
            
            # Preprocess image and move it to the appropriate device
            image_tensor = model_manager.preprocess(image).unsqueeze(0).to(model_manager.device)
            
            # Generate image embedding using the model
            embedding = model_manager.encode_image(image_tensor)
            
            # Move embedding to CPU and convert to NumPy array
            embeddings_list.append(embedding.cpu().numpy())
        
        # Stack all embeddings into a single NumPy array
        self.image_embeddings = np.vstack(embeddings_list)
        
        # Save embeddings to disk
        np.save(settings.EMBEDDINGS_PATH, self.image_embeddings)
        print(f"💾 Embeddings saved. Shape: {self.image_embeddings.shape}")
    
    def _build_faiss_index(self) -> None:
        """Build and save FAISS index for similarity search"""
        # Determine embedding vector dimension
        embedding_dim = self.image_embeddings.shape[1]
        
        # Initialize FAISS index using Inner Product (for cosine similarity)
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Add image embeddings to the FAISS index
        self.index.add(self.image_embeddings)
        
        # Log index creation
        print(f"✅ FAISS index built with {self.index.ntotal} vectors")
        
        # Save FAISS index to disk
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
        # Perform similarity search on the FAISS index
        return self.index.search(query_embedding, k)


# ==============================
# Global Dataset Manager Instance
# ==============================

# Create a global DatasetManager instance for reuse across the application
dataset_manager = DatasetManager()