from __future__ import annotations

import io
import logging
from pathlib import Path

import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.schemas import Product, SearchResult

logger = logging.getLogger(__name__)

INDEX_FILE = Path("data/index.faiss")
ID_MAP_FILE = Path("data/id_map.json")
MODEL_NAME = "clip-ViT-B-32"
SIMILARITY_THRESHOLD = 0.20


class SearchService:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._id_map: list[int] = []

    def load(self, products: list[Product]) -> None:
        import json

        logger.info("Loading CLIP model...")
        self._model = SentenceTransformer(MODEL_NAME)

        if INDEX_FILE.exists() and ID_MAP_FILE.exists():
            logger.info("Loading FAISS index from disk...")
            self._index = faiss.read_index(str(INDEX_FILE))
            self._id_map = json.loads(ID_MAP_FILE.read_text())
        else:
            logger.warning("Index not found — run scripts/build_index.py first")

    @property
    def ready(self) -> bool:
        return self._model is not None and self._index is not None

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        vec = self._model.encode(image, convert_to_numpy=True)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.astype("float32").reshape(1, -1)

    def _encode_text(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, convert_to_numpy=True)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.astype("float32").reshape(1, -1)

    def _search(
        self,
        query_vec: np.ndarray,
        products_by_id: dict[int, Product],
        k: int = 8,
    ) -> list[SearchResult]:
        scores, indices = self._index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < SIMILARITY_THRESHOLD:
                continue
            product_id = self._id_map[idx]
            product = products_by_id.get(product_id)
            if product:
                results.append(SearchResult(product=product, score=float(score)))
        return results

    def search_by_image(
        self, image_bytes: bytes, products_by_id: dict[int, Product], k: int = 8
    ) -> list[SearchResult]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        vec = self._encode_image(image)
        return self._search(vec, products_by_id, k)

    def search_by_text(
        self, query: str, products_by_id: dict[int, Product], k: int = 8
    ) -> list[SearchResult]:
        vec = self._encode_text(query)
        return self._search(vec, products_by_id, k)


search_service = SearchService()
