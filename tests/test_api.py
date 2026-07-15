"""API endpoint tests using FastAPI TestClient."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_lifespan(tmp_path, monkeypatch):
    """Prevent real model loading during tests."""
    products_data = {
        "products": [
            {
                "id": 1,
                "title": "Test Phone",
                "price": 299.99,
                "category": "smartphones",
                "thumbnail": "https://example.com/phone.jpg",
                "description": "A test phone",
            },
            {
                "id": 2,
                "title": "Test Laptop",
                "price": 999.99,
                "category": "laptops",
                "thumbnail": "https://example.com/laptop.jpg",
                "description": "A test laptop",
            },
        ]
    }
    products_file = tmp_path / "products.json"
    products_file.write_text(json.dumps(products_data))
    monkeypatch.setattr("app.catalog.DATA_FILE", products_file)

    mock_service = MagicMock()
    mock_service.ready = False
    monkeypatch.setattr("app.main.search_service", mock_service)
    monkeypatch.setattr("app.search.search_service", mock_service)

    async def noop_fetch():
        pass

    monkeypatch.setattr("app.main.fetch_and_cache_products", noop_fetch)


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "products" in data


def test_products_default(client):
    resp = client.get("/api/products")
    assert resp.status_code == 200
    data = resp.json()
    assert "products" in data
    assert data["total"] >= 0


def test_products_category_filter(client):
    resp = client.get("/api/products?category=smartphones")
    assert resp.status_code == 200
    data = resp.json()
    for p in data["products"]:
        assert p["category"] == "smartphones"


def test_products_pagination(client):
    resp = client.get("/api/products?page=1&limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["products"]) <= 1
    assert data["limit"] == 1


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Visual Product Search" in resp.content


def test_image_search_not_ready(client):
    import io
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post("/api/search/image", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert resp.status_code == 503


def test_text_search_not_ready(client):
    resp = client.get("/api/search/text?q=headphones")
    assert resp.status_code == 503


def test_image_search_wrong_type(client):
    from unittest.mock import MagicMock
    import app.main as main_module

    original = main_module.search_service
    mock = MagicMock()
    mock.ready = True
    main_module.search_service = mock

    resp = client.post(
        "/api/search/image",
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 400
    main_module.search_service = original
