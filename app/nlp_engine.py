import re

import numpy as np


# ============================================================
# 1. TEXT TOKENIZATION
# ============================================================

def tokenize(text: str):
    """
    Convert legal text into the same token format used
    during model training.

    Parameters
    ----------
    text : str
        Legal clause, sentence, or paragraph.

    Returns
    -------
    list
        Cleaned tokens.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    tokens = re.findall(
        r"\b[a-z][a-z0-9]{1,}\b",
        text
    )

    return tokens


# ============================================================
# 2. CREATE CLAUSE VECTOR
# ============================================================

def create_clause_vector(
    tokens,
    word2vec_model,
    idf_map
):
    """
    Convert tokens into a single clause embedding.

    The embedding is created using:
        Word2Vec word vectors
                  +
        TF-IDF importance weights
                  ↓
        weighted average
                  ↓
        100-dimensional clause vector
    """

    vectors = []
    weights = []

    for token in tokens:

        if (
            token in word2vec_model.wv
            and token in idf_map
        ):
            vectors.append(
                word2vec_model.wv[token]
            )

            weights.append(
                idf_map[token]
            )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------
    # If no token has an available TF-IDF weight,
    # use available Word2Vec vectors with equal weight.

    if not vectors:

        for token in tokens:

            if token in word2vec_model.wv:

                vectors.append(
                    word2vec_model.wv[token]
                )

                weights.append(1.0)

    # --------------------------------------------------------
    # No known words
    # --------------------------------------------------------
    if not vectors:

        return np.zeros(
            word2vec_model.vector_size,
            dtype=np.float32
        )

    vector_array = np.asarray(
        vectors,
        dtype=np.float32
    )

    weight_array = np.asarray(
        weights,
        dtype=np.float32
    ).reshape(-1, 1)

    weighted_vector = (
        vector_array * weight_array
    ).sum(axis=0) / max(
        weight_array.sum(),
        1e-8
    )

    return weighted_vector.astype(
        np.float32
    )


# ============================================================
# 3. SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int,
    word2vec_model,
    idf_map,
    train_embeddings,
    clause_texts,
    clause_types
):
    """
    Find the most semantically similar legal clauses.

    Parameters
    ----------
    query : str
        User's legal clause/query.

    top_k : int
        Number of results to return.

    word2vec_model
        Trained Gensim Word2Vec model.

    idf_map : dict
        TF-IDF IDF weights learned during training.

    train_embeddings : numpy.ndarray
        Normalized embeddings of indexed legal clauses.

    clause_texts : list
        Original legal clause text.

    clause_types : list
        Clause categories/types.

    Returns
    -------
    list
        Ranked semantic search results.
    """

    # --------------------------------------------------------
    # Step 1: Tokenize query
    # --------------------------------------------------------

    tokens = tokenize(query)

    # --------------------------------------------------------
    # Step 2: Convert query into an embedding
    # --------------------------------------------------------

    query_vector = create_clause_vector(
        tokens,
        word2vec_model,
        idf_map
    )

    # --------------------------------------------------------
    # Step 3: Normalize query vector
    # --------------------------------------------------------

    query_norm = query_vector / max(
        np.linalg.norm(query_vector),
        1e-8
    )

    # --------------------------------------------------------
    # Step 4: Calculate similarity
    # --------------------------------------------------------
    # Stored clause embeddings are already normalized.
    # Therefore, dot product gives cosine similarity.

    scores = train_embeddings @ query_norm

    # --------------------------------------------------------
    # Step 5: Find Top-K clauses
    # --------------------------------------------------------

    top_k = min(
        top_k,
        len(scores)
    )

    top_indices = np.argsort(
        -scores
    )[:top_k]

    # --------------------------------------------------------
    # Step 6: Build response
    # --------------------------------------------------------

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        results.append({
            "rank": rank,
            "similarity_score": round(
                float(scores[index]),
                6
            ),
            "clause_type": clause_types[index],
            "clause_text": clause_texts[index]
        })

    return results