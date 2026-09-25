# Legal Clause Similarity Engine

> A semantic search system for contracts, agreements, and policy
> documents using Word2Vec, TF-IDF weighted embeddings, and cosine
> similarity.

## Live Demo



## Overview

The **Legal Clause Similarity Engine** accepts a legal clause, sentence,
or short paragraph and retrieves the most semantically similar clauses
from a predefined legal document dataset.

Instead of relying only on exact keyword matching, the system represents
legal text as numerical vectors and compares their semantic similarity.

## Problem Statement

Legal and compliance teams often work with large collections of
contracts, agreements, policies, terms and conditions, and compliance
documents. Finding similar clauses manually can be time-consuming,
especially when the same concept is expressed using different wording.

This project builds a reusable semantic retrieval system that:

1.  Accepts legal clauses, sentences, or short paragraphs.
2.  Preprocesses the input text.
3.  Converts legal text into numerical representations.
4.  Calculates semantic similarity.
5.  Returns the Top-K most similar clauses.
6.  Ranks relevant clauses above unrelated clauses.
7.  Generalizes reasonably to previously unseen legal text.

## Key Features

-   Semantic clause retrieval
-   Word2Vec word embeddings
-   TF-IDF weighted clause embeddings
-   Cosine similarity search
-   Top-K ranked results
-   Similarity score for every result
-   Clause-type information in search results
-   FastAPI REST API
-   Interactive browser-based UI
-   Reusable NLP engine separated from API layer
-   Saved model artifacts for inference
-   Health-check endpoint
-   Responsive desktop/mobile interface
-   Render deployment ready

## Technology Stack

| Area | Technology |
| :--- | :--- |
| **Language** | Python 3.13 |
| **NLP** | Gensim Word2Vec |
| **Text representation** | TF-IDF weighted Word2Vec embeddings |
| **Similarity** | Cosine similarity |
| **ML utilities** | scikit-learn |
| **Data processing** | Pandas, NumPy |
| **Model persistence** | Joblib |
| **Backend** | FastAPI |
| **Server** | Uvicorn |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Render |
| **Version control** | Git / GitHub |

## Architecture

``` text
User Legal Clause
        |
        v
Text Preprocessing
        |
        v
Tokenization
        |
        v
TF-IDF Weighting
        |
        v
Word2Vec Word Embeddings
        |
        v
TF-IDF Weighted Clause Vector
        |
        v
Cosine Similarity
        |
        v
Rank Indexed Clauses
        |
        v
Top-K Similar Clauses
        |
        v
FastAPI Response
        |
        v
Web UI
```

### Word2Vec configuration

The trained Word2Vec model uses:

-   Vector size: 100
-   Window: 5
-   Minimum word count: 2
-   Skip-gram: enabled
-   Negative sampling: 10
-   Epochs: 10
-   Random seed: 42

### TF-IDF weighted embeddings

TF-IDF values are used as weights when combining Word2Vec word vectors
into a single clause vector. This gives greater influence to words that
are more informative within the corpus.

### Similarity

The query vector is compared with indexed clause vectors using cosine
similarity. Results are sorted from highest similarity to lowest
similarity.

The UI displays the cosine similarity score as a percentage.

## Dataset

Dataset:

``` text
data/legal_docs.csv
```

Development dataset statistics:

-   Raw rows: 21,187
-   Missing `clause_text`: 43
-   Duplicate `clause_text`: 108
-   Clean unique clauses: 21,073
-   Clause types: 42
-   Training clauses: 16,858
-   Test clauses: 4,215

## Evaluation

The current evaluation uses `clause_type` as a **weak relevance proxy**.
A retrieved clause is treated as relevant when it belongs to the same
clause type as the query.

  Metric                          Result
  ----------------------------- --------
  Recall@1                         88.5%
  Recall@3                         93.3%
  Recall@5                         95.3%
  Recall@10                        97.0%
  Top-1 clause-type agreement      88.5%

**Important:** these figures should not be interpreted as true semantic
accuracy. `clause_type` is an indirect evaluation signal rather than
human-verified semantic relevance. A production evaluation should use
human-labeled query-to-relevant-clause pairs.

## Project Structure

``` text
legal_clause_similarity_engine/
├── app/
│   ├── app.py
│   └── nlp_engine.py
├── data/
│   └── legal_docs.csv
├── models/
│   ├── word2vec_bundle.pkl
│   ├── retrieval_index.pkl
│   └── evaluation_metrics.pkl
├── notebooks/
│   └── Legal_Clause_Similarity_Engine.ipynb
├── static/
│   ├── css/
│   │   └── style.css
│   ├── icons/
│   │   ├── github.png
│   │   ├── linkedin.png
│   │   └── legal.jpg
│   └── js/
│       └── app.js
├── templates/
│   └── index.html
├── tests/
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
└── requirements.txt
```

### Application modules

**`app/app.py`** - FastAPI application setup - Model artifact loading -
API routes - Request validation - Health endpoint - Frontend serving -
API/NLP orchestration

**`app/nlp_engine.py`** - Tokenization - Clause vector creation -
Semantic search

The NLP logic is intentionally separated from the API layer so it can be
reused and tested independently.

## Saved Model Artifacts

### `word2vec_bundle.pkl`

Contains the trained Word2Vec model and IDF information required to
generate clause vectors.

### `retrieval_index.pkl`

Contains precomputed clause embeddings and clause metadata used for
similarity search.

### `evaluation_metrics.pkl`

Contains evaluation results generated during model evaluation.

The application loads these artifacts during startup and does not
retrain the model for every request.

## API

### Health Check

``` http
GET /health
```

Example:

``` json
{
  "status": "healthy",
  "indexed_clauses": 16858,
  "vector_size": 100
}
```

### Semantic Search

``` http
POST /api/search
```

Request:

``` json
{
  "query": "The customer must make payment within thirty days.",
  "top_k": 5
}
```

Response:

``` json
{
  "query": "The customer must make payment within thirty days.",
  "results": [
    {
      "rank": 1,
      "similarity_score": 0.94,
      "clause_type": "Payment",
      "clause_text": "..."
    }
  ]
}
```

Interactive API documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

## Example Queries

### Confidentiality

``` text
The parties shall keep all confidential information secure and shall not disclose it to third parties.
```

### Payment

``` text
All outstanding amounts must be paid within thirty days from the invoice date.
```

### Termination

``` text
Either party may terminate the agreement by providing written notice.
```

### Compliance

``` text
The supplier must comply with all applicable laws and regulatory requirements.
```

## Running Locally

### 1. Clone

``` bash
git clone https://github.com/a-dinesh-kumar/legal-clause-similarity-engine.git
cd legal-clause-similarity-engine
```

### 2. Create a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

``` powershell
pip install -r requirements.txt
```

Dependencies are pinned to keep the runtime environment aligned with the
environment used to create the saved artifacts.

### 4. Start the application

``` powershell
uvicorn app.app:app --reload
```

### 5. Open

``` text
http://127.0.0.1:8000
```

API docs:

``` text
http://127.0.0.1:8000/docs
```

Health:

``` text
http://127.0.0.1:8000/health
```

## Frontend

The web interface provides:

-   Legal clause input
-   Top-K selection
-   Search action
-   Ranked results
-   Similarity percentage
-   Clause type
-   Retrieved clause text
-   GitHub and LinkedIn links
-   Responsive desktop/mobile layout

The frontend communicates with:

``` text
POST /api/search
```

## Deployment

The application is designed for deployment on Render.

Build command:

``` bash
pip install -r requirements.txt
```

Start command:

``` bash
uvicorn app.app:app --host 0.0.0.0 --port $PORT
```

The repository includes `.python-version` for Python 3.13 alignment.

The deployment loads the pre-trained artifacts from `models/`; no
training is required during deployment.

## Limitations

-   Word2Vec provides word-level representations and does not have the
    contextual understanding of modern transformer-based embedding
    models.
-   Current evaluation uses `clause_type` as a proxy rather than human
    semantic relevance labels.
-   Retrieval quality depends on the language represented in the indexed
    dataset.
-   Queries containing terminology poorly represented in the Word2Vec
    vocabulary may produce weaker representations.
-   The application is an NLP/ML demonstration and is not a substitute
    for legal review or professional legal advice.

## Future Improvements

-   Transformer-based sentence embeddings
-   Sentence-BERT or modern embedding models
-   Hybrid keyword + semantic retrieval
-   Vector database integration
-   Approximate nearest-neighbor search
-   Transformer-based reranking
-   Human-labeled semantic evaluation dataset
-   Precision@K and MRR evaluation
-   Metadata filtering
-   Batch document ingestion
-   Authentication and authorization
-   Monitoring and observability
-   Additional legal document formats

## Learning Outcomes

This project provided hands-on experience with:

-   NLP preprocessing and tokenization
-   Word2Vec
-   TF-IDF
-   Word embeddings
-   Weighted document embeddings
-   Cosine similarity
-   Semantic search
-   Information retrieval evaluation
-   Model serialization with Joblib
-   FastAPI REST API development
-   Frontend/backend integration
-   Reusable Python module design
-   Saved-artifact inference
-   Deployment preparation

## Assessment Alignment

The implementation addresses the assessment requirements to:

-   Accept legal clauses, sentences, and short paragraphs
-   Preprocess input
-   Generate embeddings
-   Calculate semantic similarity
-   Return Top-K results
-   Rank relevant clauses above unrelated clauses
-   Preserve meaning despite wording differences
-   Test generalization on unseen legal text
-   Provide a working API
-   Provide documentation
-   Demonstrate the end-to-end processing pipeline

### 👨‍💻 Author

**Dinesh Kumar Alagarsamy**  
*Software Quality Engineer / SDET*
-   GitHub: https://github.com/a-dinesh-kumar
-   LinkedIn: https://www.linkedin.com/in/dinesh-kumar-alagarsamy

#### Areas of Interest:
* Quality Engineering
* Test Automation
* API Testing
* Artificial Intelligence
* Machine Learning
* AI-powered Testing
* Software Architecture
