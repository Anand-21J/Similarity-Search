# Fashion AI Search 🛍️

AI-powered fashion similarity search using CLIP embeddings and FAISS for fast retrieval.

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
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

The application will:
- Load the CLIP model
- Download the fashion dataset (first run only)
- Generate embeddings (first run only)
- Start the web server at `http://localhost:8000`

### 3. Use the Web Interface

1. Open `http://localhost:8000` in your browser
2. Upload a fashion image
3. Get similar fashion recommendations!

## ⚙️ Configuration

Edit `config.py` or create a `.env` file to customize:

```python
# Model settings
CLIP_MODEL_NAME = "ViT-B-32"
SAMPLE_SIZE = 1000

# Search parameters
TOP_N = 30
SIMILARITY_THRESHOLD = 0.25
CLIP_WEIGHT = 0.7
COLOR_WEIGHT = 0.3

# Server settings
HOST = "0.0.0.0"
PORT = 8000

# Ngrok settings
USE_NGROK = False           # Set to True to enable ngrok
NGROK_AUTH_TOKEN = None     # Optional: for custom domains
```

## 🌐 Using Ngrok for Public Access

### Option 1: Basic Ngrok (Free)

1. Set `USE_NGROK = True` in `config.py`
2. Run the app: `python main.py`
3. Share the public URL that appears in the console

### Option 2: With Auth Token (Recommended)

1. Sign up at [ngrok.com](https://ngrok.com) (free)
2. Get your auth token from the dashboard
3. Add to `config.py`:
   ```python
   USE_NGROK = True
   NGROK_AUTH_TOKEN = "your-token-here"
   ```
4. Run the app: `python main.py`

### Benefits of Auth Token:
- ✅ Longer session times
- ✅ Custom domains (paid plans)
- ✅ More concurrent tunnels
- ✅ Better rate limits

### Environment Variable (Recommended for Security)

Create a `.env` file:
```
USE_NGROK=true
NGROK_AUTH_TOKEN=your-token-here
```

The app will automatically load these settings!

## 📦 Components

### Core Components

- **ModelManager**: Handles CLIP model loading and inference
- **DatasetManager**: Manages dataset, embeddings, and FAISS index
- **SearchService**: Implements the complete search pipeline

### API Routes

- `GET /`: Main web interface
- `POST /api/search`: Search for similar fashion items
- `GET /api/stats`: Get system statistics

### Utilities

- **image_utils**: Image processing and color analysis
- **paths**: Directory management

## 🎯 Features

- **Visual Similarity**: CLIP-based image embeddings
- **Color Matching**: Average color similarity scoring
- **Category Filtering**: Automatic category inference
- **Fast Search**: FAISS-powered vector search
- **Beautiful UI**: Modern, responsive interface

## 🔧 Advanced Usage

### Using ngrok for Public Access

Set `USE_NGROK = True` in `config.py` to create a public URL:

```python
USE_NGROK = True
NGROK_AUTH_TOKEN = "your-token"  # Optional but recommended
```

See the **Using Ngrok for Public Access** section above for details.

### Custom Dataset

Modify `DATASET_NAME` in `config.py`:

```python
DATASET_NAME = "your-username/your-dataset"
```

### Adjusting Search Parameters

Fine-tune the search behavior:

```python
TOP_N = 50  # More candidates to consider
SIMILARITY_THRESHOLD = 0.3  # Stricter matching
CLIP_WEIGHT = 0.8  # More emphasis on visual similarity
COLOR_WEIGHT = 0.2  # Less emphasis on color
```

## 📊 Performance

- **First Run**: ~10-15 minutes (downloads dataset and generates embeddings)
- **Subsequent Runs**: ~30 seconds (loads pre-computed embeddings)
- **Search Time**: <100ms per query

## 🐛 Troubleshooting

### CUDA Out of Memory

Use CPU instead:

```python
DEVICE = "cpu"
```

### Slow First Run

Reduce sample size:

```python
SAMPLE_SIZE = 500
```

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Credits

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training
- **FAISS**: Facebook AI Similarity Search
- **Dataset**: Fashion Product Images (Hugging Face)