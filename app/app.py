from pathlib import Path
import re
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.nlp_engine import semantic_search


# ============================================================
# 1. PROJECT PATHS
# ============================================================
# Resolve paths relative to the project root so the application
# works consistently in VS Code and later on Render.

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


# ============================================================
# 2. LOAD SAVED MODEL ARTIFACTS
# ============================================================
# These files were created during the Colab training phase.
#
# word2vec_bundle.pkl:
#   - trained Word2Vec model
#   - IDF weights
#   - vector size
#
# retrieval_index.pkl:
#   - normalized clause embeddings
#   - clause text
#   - clause types

WORD2VEC_BUNDLE_PATH = MODEL_DIR / "word2vec_bundle.pkl"
RETRIEVAL_INDEX_PATH = MODEL_DIR / "retrieval_index.pkl"


word2vec_bundle = joblib.load(WORD2VEC_BUNDLE_PATH)
retrieval_index = joblib.load(RETRIEVAL_INDEX_PATH)

word2vec_model = word2vec_bundle["word2vec_model"]
idf_map = word2vec_bundle["idf_map"]

train_embeddings = retrieval_index["embeddings"]
clause_texts = retrieval_index["clause_text"]
clause_types = retrieval_index["clause_type"]


# ============================================================
# 3. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Legal Clause Similarity Engine",
    description=(
        "Semantic search API for finding legally similar clauses "
        "using TF-IDF weighted Word2Vec embeddings."
    ),
    version="1.0.0"
)


# Allow frontend requests during local development and deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve CSS, JavaScript and other static assets.
app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ============================================================
# 4. REQUEST MODEL
# ============================================================

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
# 5. FRONTEND
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        TEMPLATES_DIR / "index.html"
    )


# ============================================================
# 6. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Word2Vec + TF-IDF",
        "indexed_clauses": len(clause_texts),
        "vector_size": word2vec_model.vector_size
    }


# ============================================================
# 7. SEMANTIC SEARCH API
# ============================================================

@app.post("/api/search")
def search(request: SearchRequest):

    if not request.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if request.top_k < 1 or request.top_k > 20:

        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20."
        )

    results = semantic_search(
        query=request.query,
        top_k=request.top_k,
        word2vec_model=word2vec_model,
        idf_map=idf_map,
        train_embeddings=train_embeddings,
        clause_texts=clause_texts,
        clause_types=clause_types
    )

    return {
        "query": request.query,
        "top_k": request.top_k,
        "results": results
    }