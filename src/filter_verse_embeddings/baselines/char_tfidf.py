"""Small custom char n-gram TF-IDF implementation."""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy import sparse

def iter_char_ngrams(text: str, ngram_min: int, ngram_max: int):
    for ngram_size in range(ngram_min, ngram_max + 1):
        if len(text) < ngram_size:
            continue
        for start in range(len(text) - ngram_size + 1):
            yield text[start:start + ngram_size]

def build_vocabulary_and_idf(texts, normalizer, ngram_min: int, ngram_max: int, min_ngram_doc_freq: int) -> tuple[dict[str, int], np.ndarray]:
    document_frequency: Counter[str] = Counter()
    num_docs = 0
    for text in texts:
        normalized = normalizer(text)
        unique_ngrams = set(iter_char_ngrams(normalized, ngram_min, ngram_max))
        if unique_ngrams:
            document_frequency.update(unique_ngrams)
        num_docs += 1
    vocabulary = {ngram: idx for (idx, ngram) in enumerate(sorted((ngram for (ngram, count) in document_frequency.items() if count >= min_ngram_doc_freq)))}
    idf = np.empty(len(vocabulary), dtype=np.float32)
    for (ngram, idx) in vocabulary.items():
        df = document_frequency[ngram]
        idf[idx] = np.log((1.0 + num_docs) / (1.0 + df)) + 1.0
    return (vocabulary, idf)

def l2_normalize_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    nonzero = norms > 0
    inv_norms = np.zeros_like(norms, dtype=np.float32)
    inv_norms[nonzero] = 1.0 / norms[nonzero]
    return sparse.diags(inv_norms).dot(matrix).tocsr()

def transform_texts(texts, vocabulary: dict[str, int], idf: np.ndarray, normalizer, ngram_min: int, ngram_max: int, batch_size: int) -> sparse.csr_matrix:
    idf_diag = sparse.diags(idf)
    chunk_matrices = []
    text_batch: list[str] = []

    def batch_to_matrix(batch_texts: list[str]) -> sparse.csr_matrix:
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for (row_idx, text) in enumerate(batch_texts):
            counts: Counter[int] = Counter()
            normalized = normalizer(text)
            for ngram in iter_char_ngrams(normalized, ngram_min, ngram_max):
                vocab_idx = vocabulary.get(ngram)
                if vocab_idx is not None:
                    counts[vocab_idx] += 1
            for (vocab_idx, count) in counts.items():
                rows.append(row_idx)
                cols.append(vocab_idx)
                data.append(float(count))
        matrix = sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(len(batch_texts), len(vocabulary)), dtype=np.float32)
        matrix = matrix.dot(idf_diag)
        return l2_normalize_rows(matrix)
    for text in texts:
        text_batch.append(text)
        if len(text_batch) >= batch_size:
            chunk_matrices.append(batch_to_matrix(text_batch))
            text_batch = []
    if text_batch:
        chunk_matrices.append(batch_to_matrix(text_batch))
    if not chunk_matrices:
        return sparse.csr_matrix((0, len(vocabulary)), dtype=np.float32)
    if len(chunk_matrices) == 1:
        return chunk_matrices[0]
    return sparse.vstack(chunk_matrices, format='csr')
