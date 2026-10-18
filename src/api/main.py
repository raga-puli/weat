"""FastAPI app: `uvicorn src.api.main:app` from the repo root.

Shared and frozen. Endpoints live in the routers (routes_group.py: Person A,
routes_search.py: Person B). Heavy resources are loaded lazily through
lru_cache getters used as dependencies, never at import time.
"""

from fastapi import FastAPI

from src.api import routes_group, routes_search

app = FastAPI(title="Group Restaurant Recommender")
app.include_router(routes_group.router)
app.include_router(routes_search.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
