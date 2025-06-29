from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.catalog import fetch_and_cache_products, get_categories, get_products_page, load_products
from app.schemas import Product, ProductsResponse, SearchResponse
from app.search import search_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_products: list[Product] = []
_products_by_id: dict[int, Product] = {}

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _products, _products_by_id
    await fetch_and_cache_products()
    _products = load_products()
    _products_by_id = {p.id: p for p in _products}
    search_service.load(_products)
    logger.info(f"Loaded {len(_products)} products, index ready: {search_service.ready}")
    yield


app = FastAPI(title="Visual Product Search", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "products": len(_products), "index_ready": search_service.ready}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    categories = get_categories(_products)
    return templates.TemplateResponse(
        "index.html", {"request": request, "categories": categories}
    )


@app.get("/api/products", response_model=ProductsResponse)
def list_products(
    category: str | None = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    page_products, total = get_products_page(_products, category, page, limit)
    return ProductsResponse(products=page_products, total=total, page=page, limit=limit)


@app.post("/api/search/image", response_model=SearchResponse)
async def search_by_image(file: UploadFile = File(...)):
    if not search_service.ready:
        raise HTTPException(503, "Search index not ready. Run scripts/build_index.py first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image.")
    data = await file.read()
    results = search_service.search_by_image(data, _products_by_id)
    return SearchResponse(results=results, query_type="image")


@app.get("/api/search/text", response_model=SearchResponse)
def search_by_text(q: str = Query(..., min_length=1, max_length=200)):
    """
    Natural-language search via CLIP text encoder.
    Queries are matched against product *images* — not just titles.
    """
    if not search_service.ready:
        raise HTTPException(503, "Search index not ready. Run scripts/build_index.py first.")
    results = search_service.search_by_text(q.strip(), _products_by_id)
    return SearchResponse(results=results, query_type="text")
