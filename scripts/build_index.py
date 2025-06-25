"""
Offline indexing script — run once before starting the server.
Fetches DummyJSON products, downloads images, encodes with CLIP, builds FAISS index.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import faiss
import httpx
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
IMAGES_DIR = DATA_DIR / "images"
PRODUCTS_FILE = DATA_DIR / "products.json"
INDEX_FILE = DATA_DIR / "index.faiss"
ID_MAP_FILE = DATA_DIR / "id_map.json"
MODEL_NAME = "clip-ViT-B-32"
API_URL = "https://dummyjson.com/products?limit=200&skip=0"


async def fetch_products() -> list[dict]:
    if PRODUCTS_FILE.exists():
        logger.info("Using cached products.json")
        return json.loads(PRODUCTS_FILE.read_text())["products"]

    logger.info("Fetching products from DummyJSON…")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(API_URL)
        resp.raise_for_status()
        data = resp.json()

    PRODUCTS_FILE.write_text(json.dumps(data, indent=2))
    logger.info(f"Saved {len(data['products'])} products to {PRODUCTS_FILE}")
    return data["products"]


async def download_images(products: list[dict]) -> dict[int, Path]:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    id_to_path: dict[int, Path] = {}

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for p in products:
            pid = p["id"]
            url = p.get("thumbnail", "")
            dest = IMAGES_DIR / f"{pid}.jpg"

            if dest.exists():
                id_to_path[pid] = dest
                continue

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                id_to_path[pid] = dest
            except Exception as e:
                logger.warning(f"Failed to download image for product {pid}: {e}")

    logger.info(f"Downloaded {len(id_to_path)} images")
    return id_to_path


def build_index(products: list[dict], id_to_path: dict[int, Path]) -> None:
    logger.info("Loading CLIP model…")
    model = SentenceTransformer(MODEL_NAME)

    vectors: list[np.ndarray] = []
    id_map: list[int] = []

    for p in products:
        pid = p["id"]
        path = id_to_path.get(pid)
        if not path:
            continue

        try:
            img = Image.open(path).convert("RGB")
            vec = model.encode(img, convert_to_numpy=True)
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            vectors.append(vec.astype("float32"))
            id_map.append(pid)
        except Exception as e:
            logger.warning(f"Failed to encode product {pid}: {e}")

    if not vectors:
        logger.error("No vectors produced — aborting")
        return

    matrix = np.stack(vectors)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product = cosine on normalized vecs
    index.add(matrix)

    faiss.write_index(index, str(INDEX_FILE))
    ID_MAP_FILE.write_text(json.dumps(id_map))

    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}")
    logger.info(f"Saved → {INDEX_FILE}, {ID_MAP_FILE}")


async def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    products = await fetch_products()
    id_to_path = await download_images(products)
    build_index(products, id_to_path)
    logger.info("Done. Start the server with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
