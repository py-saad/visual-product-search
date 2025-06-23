import json
import httpx
from pathlib import Path
from app.schemas import Product

DATA_FILE = Path("data/products.json")
DUMMYJSON_URL = "https://dummyjson.com/products?limit=200&skip=0"


async def fetch_and_cache_products() -> list[dict]:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text())["products"]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(DUMMYJSON_URL)
        resp.raise_for_status()
        data = resp.json()

    DATA_FILE.write_text(json.dumps(data, indent=2))
    return data["products"]


def load_products() -> list[Product]:
    if not DATA_FILE.exists():
        return []
    raw = json.loads(DATA_FILE.read_text())["products"]
    return [Product(**p) for p in raw]


def get_products_page(
    products: list[Product],
    category: str | None = None,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[Product], int]:
    filtered = products
    if category:
        filtered = [p for p in products if p.category == category]

    total = len(filtered)
    start = (page - 1) * limit
    return filtered[start : start + limit], total


def get_categories(products: list[Product]) -> list[str]:
    return sorted({p.category for p in products})
