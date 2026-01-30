
## Overview

Fashion AI Search is a visual search engine that finds similar fashion products using deep learning and computer vision. Users can upload an image of a fashion item, and the system returns visually similar products from a curated dataset.

### Key Features
- **CLIP-based Image Embedding**: Uses OpenAI's CLIP model for semantic understanding
- **Color-aware Matching**: Combines visual features with color similarity
- **Category Filtering**: Automatically infers product category for better results
- **Fast Search**: FAISS indexing for efficient nearest neighbor search
- **REST API**: FastAPI-based web service with file upload support
- **Public Access**: Ngrok integration for easy sharing

---

## 📁 Project Structure

```
fashion-ai-search/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (optional)
│
├── api/
│   ├── __init__.py
│   └── routes.py          # API endpoints
│
├── core/
│   ├── __init__.py
│   ├── model_manager.py   # CLIP model management
│   ├── dataset_manager.py # Dataset and FAISS index management
│   └── startup.py         # Application startup handler
│
├── services/
│   ├── __init__.py
│   └── search_service.py  # Search pipeline logic
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py     # Image processing utilities
│   └── paths.py           # Path management utilities
│
├── templates/
│   └── index.html         # Frontend interface
│
├── data/                  # Dataset metadata (auto-created)
└── artifacts/             # Embeddings and FAISS index (auto-created)

# Fashion AI Search - Technical Documentation

## Dependencies

### Core Libraries
```python
# Deep Learning & Computer Vision
torch                    # PyTorch for model execution
open_clip               # CLIP model implementation
numpy                   # Numerical operations
PIL (Pillow)           # Image processing

# Vector Search
faiss-cpu / faiss-gpu  # Efficient similarity search

# Dataset
datasets               # Hugging Face datasets

# Web Framework
fastapi                # Modern web framework
uvicorn                # ASGI server
jinja2                 # Template rendering

# Utilities
pyngrok                # Public URL tunneling
nest_asyncio          # Async support for notebooks
```

### Installation Command
```bash
pip install torch open-clip-torch faiss-cpu datasets fastapi uvicorn jinja2 pyngrok nest-asyncio pillow numpy
```

---

## Configuration

### Global Constants

```python
# Dataset Configuration
SAMPLE_SIZE = 1000              # Number of images to index
TOP_N = 30                      # Initial candidates to retrieve
SIMILARITY_THRESHOLD = 0.25     # Minimum CLIP score threshold

# Scoring Weights
CLIP_WEIGHT = 0.7              # Weight for visual similarity
COLOR_WEIGHT = 0.3             # Weight for color similarity

# Directory Structure
DATA_DIR = Path("data")                    # Dataset metadata
ARTIFACTS_DIR = Path("artifacts")          # Model artifacts
TEMPLATES_DIR = Path("templates")          # HTML templates

# Artifact Paths
FAISS_INDEX_PATH = "artifacts/fashion_faiss.index"
EMBEDDINGS_PATH = "artifacts/image_embeddings.npy"
METADATA_PATH = "data/sampled_dataset_info.json"
```

---

## System Components

### 1. CLIP Model Loading

**Function**: `load_clip_model()`

- **Model**: ViT-B-32 (Vision Transformer, Base, 32×32 patches)
- **Pretrained Weights**: OpenAI's official weights
- **Purpose**: Converts images to 512-dimensional embeddings
- **Device**: Automatically uses GPU if available, falls back to CPU

### 2. Dataset Management

**Function**: `load_or_create_dataset()`

- **Dataset Source**: `ashraq/fashion-product-images-small` from Hugging Face
- **Behavior**:
  - Checks for existing embeddings/index
  - Loads pre-computed artifacts if available
  - Creates new embeddings if not found
- **Persistence**: All artifacts saved to disk for quick restart

**Metadata Fields**:
```json
{
  "articleType": "Shirts",
  "baseColour": "Blue",
  "season": "Summer",
  "usage": "Casual",
  "gender": "Men",
  "masterCategory": "Apparel",
  "subCategory": "Topwear",
  "productDisplayName": "Men Blue Shirt"
}
```

### 3. Embedding Generation

**Function**: `generate_embeddings()`

**Process**:
1. Iterate through all sampled images
2. Preprocess each image (resize, normalize)
3. Generate CLIP embedding (512-dim vector)
4. Normalize to unit length (for cosine similarity)
5. Save to NumPy array

**Output**: `image_embeddings.npy` - Shape: (1000, 512)

### 4. FAISS Indexing

**Function**: `build_faiss_index()`

- **Index Type**: `IndexFlatIP` (Inner Product for cosine similarity)
- **Similarity Metric**: Cosine similarity (via normalized vectors)
- **Size**: 1000 vectors × 512 dimensions
- **Search Complexity**: O(n) - exhaustive search for accuracy

### 5. Image Processing Helpers

**Color Analysis**:
```python
get_avg_color(image) -> np.ndarray
# Returns mean RGB values [R, G, B]

color_similarity(c1, c2) -> float
# Returns normalized color distance (0-1)
# Formula: max(0, 1 - L2_distance / 255)
```

**Image Encoding**:
```python
image_to_base64(image) -> str
# Converts PIL Image to data URI
# Format: "data:image/png;base64,{encoded_data}"
```

---

## API Endpoints

### 1. GET `/`

**Purpose**: Serve main HTML interface

**Response**: HTML page with:
- Upload interface
- System statistics
- Results display area

**Template Variables**:
```python
{
    "sample_size": 1000,
    "embedding_dim": 512,
    "index_size": 1000,
    "device": "CUDA"
}
```

### 2. POST `/api/search`

**Purpose**: Find similar fashion items

**Request**:
```
Content-Type: multipart/form-data

file: <image_file>      # Required, image file
top_k: 5                # Optional, number of results (default: 5)
```

**Processing Steps**:
1. Read uploaded image
2. Generate CLIP embedding
3. Search FAISS index for top 30 candidates
4. Extract color features
5. Infer query category from top results
6. Filter by category
7. Calculate combined scores
8. Rank and return top-k results

**Response**:
```json
{
  "success": true,
  "query_category": "Shirts",
  "results": [
    {
      "rank": 1,
      "name": "Men Blue Casual Shirt",
      "image": "data:image/png;base64,...",
      "percentage": 87,
      "details": {
        "CLIP Score": "0.892",
        "Color Score": "0.834",
        "Final Score": "0.874",
        "Category": "Shirts",
        "Article Type": "Shirts",
        "Color": "Blue",
        "Season": "Summer",
        "Usage": "Casual",
        "Gender": "Men",
        "Master Category": "Apparel",
        "Sub Category": "Topwear"
      }
    }
  ]
}
```

### 3. GET `/api/stats`

**Purpose**: Get system statistics

**Response**:
```json
{
  "sample_size": 1000,
  "embedding_dim": 512,
  "index_size": 1000,
  "device": "cuda"
}
```

---

## Search Algorithm

### Multi-Stage Ranking Process

#### Stage 1: FAISS Retrieval
```python
scores, indices = index.search(query_embedding, TOP_N=30)
# Returns top 30 candidates by cosine similarity
```

#### Stage 2: Threshold Filtering
```python
if clip_score < SIMILARITY_THRESHOLD (0.25):
    skip_candidate
```

#### Stage 3: Category Inference
```python
# Infer query category from top 10 results
candidate_categories = [top_10_results.categories]
query_category = most_common(candidate_categories)
```

#### Stage 4: Category Filtering
```python
if query_category and candidate_category != query_category:
    skip_candidate
```

#### Stage 5: Score Combination
```python
clip_score = cosine_similarity(query_emb, candidate_emb)
color_score = color_similarity(query_color, candidate_color)

final_score = (0.7 × clip_score) + (0.3 × color_score)
```

#### Stage 6: Final Ranking
```python
sorted_results = sort_by(final_score, descending=True)[:top_k]
```

### Scoring Formula

**CLIP Score**: Semantic similarity (0-1)
- Based on deep visual features
- Captures shape, texture, style

**Color Score**: Color similarity (0-1)
- Based on average RGB distance
- Normalized by max possible distance (255)

**Final Score**:
```
final_score = 0.7 × clip_score + 0.3 × color_score
```

---

## Installation & Setup

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch open-clip-torch faiss-cpu datasets fastapi uvicorn jinja2 pyngrok nest-asyncio pillow numpy tqdm
```

### Step 2: Directory Structure
```
project/
├── app.py                          # Main application
├── templates/
│   └── index.html                  # Frontend template
├── data/                           # Created automatically
│   └── sampled_dataset_info.json
├── artifacts/                      # Created automatically
│   ├── fashion_faiss.index
│   └── image_embeddings.npy
```

### Step 3: First Run
```bash
python app.py
```

**First Run Behavior**:
1. Downloads CLIP model (~350MB)
2. Downloads fashion dataset (~500MB)
3. Generates embeddings (~5-10 minutes on CPU)
4. Builds FAISS index
5. Starts server with ngrok tunnel

**Subsequent Runs**:
- Loads pre-computed artifacts (~2-3 seconds)
- Immediately ready to serve requests

---

## Usage

### Starting the Server

```bash
python app.py
```

**Output**:
```
Using device: cuda
Loading CLIP model...
CLIP model loaded successfully
Found existing embeddings and index. Loading...
Loaded 1000 samples with 1000 vectors
🚀 Starting ngrok tunnel...
✅ Public URL: https://xxxx-xx-xx-xx-xx.ngrok.io
🌐 Ngrok Dashboard: http://127.0.0.1:4040
📱 Share this link to access your app: https://xxxx-xx-xx-xx-xx.ngrok.io
------------------------------------------------------------
🔥 Starting FastAPI server on port 8000...
```

### Using the API

**Python Example**:
```python
import requests

url = "http://localhost:8000/api/search"
files = {"file": open("my_shirt.jpg", "rb")}
data = {"top_k": 5}

response = requests.post(url, files=files, data=data)
results = response.json()

for result in results["results"]:
    print(f"Rank {result['rank']}: {result['name']} - {result['percentage']}% match")
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/search" \
  -F "file=@my_shirt.jpg" \
  -F "top_k=5"
```

---

## Technical Details

### Performance Characteristics

**Embedding Generation**:
- Time: ~100ms per image (GPU) / ~500ms (CPU)
- Memory: ~2GB VRAM (GPU) / ~4GB RAM (CPU)

**Search Performance**:
- Index build: O(n) where n = dataset size
- Search: O(n) per query (exhaustive search)
- Typical query time: ~50ms for 1000 images

**Scalability**:
- Current: 1000 images
- Recommended max (IndexFlatIP): ~100K images
- For larger datasets: Use IndexIVFFlat or IndexHNSW

### Model Details

**CLIP ViT-B/32**:
- Architecture: Vision Transformer
- Input: 224×224 RGB images
- Output: 512-dimensional embeddings
- Training: 400M image-text pairs
- Parameters: ~151M

### Memory Requirements

**Runtime Memory**:
- CLIP Model: ~600MB
- FAISS Index: ~2MB (1000 × 512 × 4 bytes)
- Image Embeddings: ~2MB
- Total: ~1GB minimum

**Storage**:
- Model weights: ~350MB
- Dataset cache: ~500MB
- Embeddings: ~2MB
- Index: ~2MB
- Total: ~900MB

---

## Error Handling

### Common Errors

**1. CUDA Out of Memory**:
```python
# Solution: Use CPU
device = "cpu"
```

**2. Dataset Download Failure**:
```python
# Solution: Manual download or retry
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
```

**3. ngrok Connection Issues**:
```python
# Solution: Check ngrok installation
pip install pyngrok
# Or disable ngrok and use localhost
```

**4. File Upload Size Limit**:
```python
# Increase in FastAPI
app = FastAPI(max_upload_size=10_000_000)  # 10MB
```

---

## Future Enhancements

### Potential Improvements

1. **Advanced Indexing**:
   - Use IndexIVFFlat for faster search
   - Implement GPU-accelerated FAISS

2. **Multi-modal Search**:
   - Add text queries ("red summer dress")
   - Combine image + text search

3. **Better Filtering**:
   - Price range filters
   - Brand filtering
   - Style preferences

4. **Production Features**:
   - Redis caching for embeddings
   - Batch processing
   - Load balancing
   - Database integration

5. **UI Enhancements**:
   - Grid view results
   - Filter controls
   - Save favorites
   - Search history

---

## License & Attribution

- **Dataset**: [Fashion Product Images (Small)](https://huggingface.co/datasets/ashraq/fashion-product-images-small)
- **CLIP Model**: OpenAI (MIT License)
- **FAISS**: Meta AI Research (MIT License)

---

## Support

For issues or questions:
1. Check FAISS documentation: https://github.com/facebookresearch/faiss
2. CLIP documentation: https://github.com/openai/CLIP
3. FastAPI documentation: https://fastapi.tiangolo.com/

---

**Last Updated**: January 2026
**Version**: 1.0.0
