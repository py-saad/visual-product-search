# Visual Product Search

An eCommerce-style product catalog with **AI-powered image search**: browse ~200 products or upload a photo to find visually similar items using CLIP embeddings + FAISS.

> **Live demo:** *(deploy link coming soon)*

---

## What it does

- **Product listing** — grid of ~200 products with category filter and pagination
- **Image search** — upload any photo → top 8 visually similar products ranked by cosine similarity
- **Text search** — natural-language queries matched against product *images* via CLIP's shared embedding space (not just title keywords)

## Tech stack

| Layer | Choice |
|---|---|
| Model | CLIP ViT-B/32 via `sentence-transformers` |
| Vector search | FAISS (IndexFlatIP, L2-normalized = cosine similarity) |
| Backend | FastAPI + Pydantic |
| Frontend | Jinja2 + vanilla JS |
| Data | DummyJSON products API (~200 products) |
| Packaging | Docker + docker-compose |
| CI | GitHub Actions — ruff lint + pytest |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the CLIP index (run once — downloads ~350 MB model + product images)
python scripts/build_index.py

# 3. Start the server
uvicorn app.main:app --reload
# → http://localhost:8000
```

### Docker

```bash
docker-compose up --build
# Then run the indexer inside the container:
docker-compose exec app python scripts/build_index.py
```

## API

```
GET  /                      product listing page
GET  /api/products          JSON catalog  ?category= &page= &limit=
POST /api/search/image      multipart image → top-k similar products
GET  /api/search/text?q=    text query → top-k products via CLIP text encoder
GET  /health                liveness + readiness check
```

Auto-generated Swagger docs at `/docs`.

## Architecture

**Offline (build once):**
1. Fetch DummyJSON catalog → `data/products.json`
2. Download product thumbnails → `data/images/`
3. Encode each image with CLIP → 512-dim L2-normalized vectors
4. Build FAISS `IndexFlatIP` → `data/index.faiss` + `data/id_map.json`

**Online (per request):**
1. Startup: load CLIP model + FAISS index into memory once (cold ~2 s, warm search < 200 ms)
2. Upload → encode with CLIP → `index.search(k=8)` → return ranked products
3. Text query uses the same path via CLIP's text encoder

### Key design notes

- Vectors are **L2-normalized** before indexing so inner-product search equals cosine similarity
- Model + index load **once at startup** as singletons — not per request
- Index building is **decoupled from serving** — reindex without redeploying the app
- A **similarity threshold** (default 0.20) filters garbage matches when the photo is unlike anything in the catalog

## Roadmap

- [ ] Swap FAISS → pgvector on PostgreSQL
- [ ] Client-side image crop before search ("search by region")
- [ ] Hybrid ranking: combine image similarity + category/price filters
- [ ] Batch re-indexing endpoint behind an API key

## Author

[Saad Ashraf](https://github.com/py-saad)
