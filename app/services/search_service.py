"""
Search Service for Fashion Similarity Search
Handles the complete search pipeline from image to results
"""

import io
import numpy as np
from PIL import Image
from typing import List, Dict

from config import settings
from core.model_manager import ModelManager
from core.dataset_manager import DatasetManager
from utils.image_utils import get_avg_color, color_similarity, image_to_base64

class SearchService:
    """Service for performing fashion similarity search"""
    
    def __init__(self, model_manager: ModelManager, dataset_manager: DatasetManager):
        """
        Initialize SearchService
        
        Args:
            model_manager: Initialized ModelManager instance
            dataset_manager: Initialized DatasetManager instance
        """
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
    
    async def search(self, image_bytes: bytes, top_k: int = 5) -> Dict:
        """
        Perform similarity search for a query image
        
        Args:
            image_bytes: Raw image bytes
            top_k: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        # Load and process query image
        query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query_image)
        
        # Search using FAISS
        scores, indices = self.dataset_manager.search(
            query_embedding,
            k=settings.TOP_N
        )
        
        # Get query color for color-based filtering
        query_color = get_avg_color(query_image)
        
        # Infer query category from top candidates
        query_category = self._infer_category(indices[0])
        
        # Rank and filter results
        ranked_results = self._rank_results(
            scores[0],
            indices[0],
            query_color,
            query_category
        )

        # Format results
        formatted_results = self._format_results(ranked_results[:top_k])

        # 🔥 FINAL GUARANTEE CHECK
        if not formatted_results:
            return {
                "success": False,
                "message": "NO SIMILAR IMAGES FOUND",
                "query_category": query_category,
                "results": []
            }

        return {
            "success": True,
            "query_category": query_category,
            "results": formatted_results
}
    
    def _generate_query_embedding(self, query_image: Image.Image) -> np.ndarray:
        """
        Generate normalized embedding for query image
        
        Args:
            query_image: PIL Image object
            
        Returns:
            Normalized embedding as numpy array
        """
        # Preprocess image
        query_tensor = self.model_manager.preprocess(query_image).unsqueeze(0).to(
            self.model_manager.device
        )
        
        # Generate embedding
        query_embedding = self.model_manager.encode_image(query_tensor)
        
        return query_embedding.cpu().numpy()
    
    def _infer_category(self, indices: np.ndarray) -> str:
        """
        Infer query category from top candidate matches
        
        Args:
            indices: Array of candidate indices
            
        Returns:
            Most common category among top candidates
        """
        # Get categories from top 10 candidates
        candidate_categories = [
            self.dataset_manager.sampled_dataset[int(idx)].get("articleType")
            for idx in indices[:10]
            if self.dataset_manager.sampled_dataset[int(idx)].get("articleType")
        ]
        
        if candidate_categories:
            # Return most common category
            return max(set(candidate_categories), key=candidate_categories.count)
        
        return None
    
    def _rank_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        query_color: np.ndarray,
        query_category: str
    ) -> List[Dict]:
        """
        Rank and filter search results
        
        Args:
            scores: CLIP similarity scores
            indices: Candidate indices
            query_color: Average RGB color of query image
            query_category: Inferred category of query
            
        Returns:
            List of ranked result dictionaries
        """
        ranked_results = []
        
        for clip_score, idx in zip(scores, indices):
            # Filter by similarity threshold
            if clip_score < settings.SIMILARITY_THRESHOLD:
                continue
            
            candidate = self.dataset_manager.sampled_dataset[int(idx)]
            candidate_category = candidate.get("articleType")
            
            # Category filter (if category was inferred)
            if query_category and candidate_category != query_category:
                continue
            
            # Calculate color similarity
            candidate_image = candidate["image"]
            candidate_color = get_avg_color(candidate_image)
            col_sim = color_similarity(query_color, candidate_color)
            
            # Calculate final weighted score
            final_score = (
                settings.CLIP_WEIGHT * float(clip_score) +
                settings.COLOR_WEIGHT * col_sim
            )
            
            ranked_results.append({
                "idx": int(idx),
                "clip_score": float(clip_score),
                "color_score": col_sim,
                "final_score": final_score,
                "category": candidate_category
            })
        
        # Sort by final score (descending)
        ranked_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return ranked_results
    
    def _format_results(self, ranked_results: List[Dict]) -> List[Dict]:
        """
        Format results for frontend display
        
        Args:
            ranked_results: List of ranked result dictionaries
            
        Returns:
            List of formatted result dictionaries
        """
        formatted_results = []
        
        for rank, result in enumerate(ranked_results, start=1):
            candidate = self.dataset_manager.sampled_dataset[result["idx"]]
            
            # Build details dictionary
            details = {
                "CLIP Score": f"{result['clip_score']:.3f}",
                "Color Score": f"{result['color_score']:.3f}",
                "Final Score": f"{result['final_score']:.3f}",
                "Category": result["category"] or "Unknown"
            }
            
            # Add additional product metadata if available
            metadata_fields = [
                ("articleType", "Article Type"),
                ("baseColour", "Color"),
                ("season", "Season"),
                ("usage", "Usage"),
                ("gender", "Gender"),
                ("masterCategory", "Master Category"),
                ("subCategory", "Sub Category")
            ]
            
            for field, label in metadata_fields:
                value = candidate.get(field)
                if value:
                    details[label] = value
            
            # Format result
            formatted_results.append({
                "rank": rank,
                "name": candidate.get("productDisplayName", "Fashion Item"),
                "image": image_to_base64(candidate["image"]),
                "percentage": int(result["final_score"] * 100),
                "details": details
            })
        
        return formatted_results

