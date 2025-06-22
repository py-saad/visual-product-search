from pydantic import BaseModel
from typing import Optional


class Product(BaseModel):
    id: int
    title: str
    price: float
    category: str
    thumbnail: str
    description: Optional[str] = None


class SearchResult(BaseModel):
    product: Product
    score: float


class ProductsResponse(BaseModel):
    products: list[Product]
    total: int
    page: int
    limit: int


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_type: str
