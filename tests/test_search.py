"""Search service unit tests — sanity checks without loading the real model."""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.schemas import Product, SearchResult
from app.search import SearchService, SIMILARITY_THRESHOLD


def _make_product(pid: int, title: str = "Product", category: str = "test") -> Product:
    return Product(id=pid, title=title, price=9.99, category=category, thumbnail="")


def _make_service_with_index(n: int = 5) -> tuple[SearchService, dict[int, Product]]:
    """Return a SearchService with a synthetic FAISS index of n items."""
    import faiss

    dim = 512
    vecs = np.random.randn(n, dim).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    products = {i + 1: _make_product(i + 1) for i in range(n)}
    id_map = list(range(1, n + 1))

    service = SearchService()
    service._index = index
    service._id_map = id_map
    service._model = MagicMock()

    return service, products, vecs


def test_same_image_ranks_first():
    """Encoding the same vector should return itself as the top result."""
    service, products, vecs = _make_service_with_index(10)

    query_vec = vecs[0].reshape(1, -1)
    service._model.encode.return_value = query_vec.flatten()

    with patch.object(service, "_encode_image", return_value=query_vec):
        fake_bytes = b"fake"
        results = service.search_by_image(fake_bytes, products)

    assert len(results) > 0
    assert results[0].product.id == 1
    assert results[0].score >= SIMILARITY_THRESHOLD


def test_threshold_filters_low_scores():
    """Results below the similarity threshold should be excluded."""
    import faiss

    dim = 512
    service = SearchService()

    # Build index with one vector pointing in a very different direction
    base = np.zeros((1, dim), dtype="float32")
    base[0, 0] = 1.0
    orthogonal = np.zeros((1, dim), dtype="float32")
    orthogonal[0, 1] = 1.0  # score = 0.0

    index = faiss.IndexFlatIP(dim)
    index.add(base)
    service._index = index
    service._id_map = [1]
    service._model = MagicMock()

    products = {1: _make_product(1)}

    with patch.object(service, "_encode_image", return_value=orthogonal):
        results = service.search_by_image(b"fake", products)

    assert all(r.score >= SIMILARITY_THRESHOLD for r in results)


def test_ready_false_before_load():
    service = SearchService()
    assert not service.ready


def test_ready_true_after_index_set():
    import faiss

    service = SearchService()
    service._model = MagicMock()
    service._index = faiss.IndexFlatIP(512)
    assert service.ready


def test_search_top_k():
    """Only up to k results should be returned."""
    service, products, vecs = _make_service_with_index(10)
    query_vec = vecs[0].reshape(1, -1)

    with patch.object(service, "_encode_text", return_value=query_vec):
        results = service.search_by_text("shoes", products, k=3)

    assert len(results) <= 3
