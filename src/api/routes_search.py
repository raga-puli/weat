"""Search endpoints (Person B).

    POST /search          text -> ParsedQuery -> ranked restaurants, with timings
    POST /search/parse    text -> ParsedQuery only (for debugging the UI)
    GET  /search/anchors  the named places `near` can refer to

engine="llm" falls back to the rules parser when Ollama is down or its output is invalid; engine_used
says which one actually ran. Closed restaurants are excluded unless include_closed=true.

Heavy resources come from lru_cache getters (get_db_conn, get_encoder, get_search_data) behind
dependencies, so tests can swap them with app.dependency_overrides. The DB connection and the parquet
frame are handed out as zero-argument providers so backend=numpy never connects to Postgres and
backend=sql never reads the parquet. At server start (the router's lifespan, merged into the app by
include_router) the encoder and DB connection are warmed and Ollama starts loading its model, so the
first search doesn't pay ~50 s of torch import; TestClient without `with` skips this.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import lru_cache
from typing import Annotated, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src import search
from src.query_parser import ANCHORS, ParsedQuery, parse_with_engine, preload_llm

log = logging.getLogger(__name__)



def _warm_up() -> None:
    threading.Thread(target=preload_llm, daemon=True).start()  # engine=llm falls back until it's ready
    start = time.perf_counter()
    get_encoder().encode("warm up", normalize_embeddings=True)
    try:
        get_db_conn()
    except (psycopg.OperationalError, RuntimeError) as err:
        log.warning("search warm-up: Postgres unavailable (%s); backend=sql will return 503 until it is", err)
    log.warning("search warm-up done in %.1fs", time.perf_counter() - start)


@asynccontextmanager
async def _lifespan(app):
    await asyncio.to_thread(_warm_up)
    yield


router = APIRouter(prefix="/search", tags=["search"], lifespan=_lifespan)

DB_CONNECT_TIMEOUT_S = 5
DB_UNAVAILABLE = "Database unavailable. Start Postgres with `docker compose up -d`, or retry with backend=numpy."


# ---------- dependencies ----------


@lru_cache(maxsize=1)
def get_db_conn() -> psycopg.Connection:
    from src.db import get_conn

    conn = get_conn(connect_timeout=DB_CONNECT_TIMEOUT_S)
    conn.autocommit = True  # read-only queries; don't leave the connection idle in a transaction
    return conn


@lru_cache(maxsize=1)
def get_encoder():
    return search.load_encoder()


@lru_cache(maxsize=1)
def get_search_data():
    return search.load_frame()


def db_conn_provider() -> Callable[[], psycopg.Connection]:
    return get_db_conn


def search_data_provider() -> Callable[[], tuple]:
    return get_search_data


def get_parser() -> Callable[[str, str], tuple[ParsedQuery, str]]:
    return parse_with_engine


# ---------- models ----------

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
Engine = Literal["rules", "llm"]


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: QueryText
    engine: Engine = "llm"


class SearchRequest(ParseRequest):
    k: int = Field(10, ge=1, le=50)
    backend: Literal["sql", "numpy"] = "sql"
    include_closed: bool = False


class ParsedQueryOut(BaseModel):
    max_price_level: int | None
    near: str | None
    max_distance_km: float | None
    dietary: list[str]
    open_at: str | None
    categories: list[str]
    soft: str


class ParseResponse(BaseModel):
    parsed: ParsedQueryOut
    engine_used: Engine


class Restaurant(BaseModel):
    business_id: str
    name: str
    categories: list[str]
    price_level: int | None
    lat: float | None
    lon: float | None
    distance_km: float | None
    similarity: float | None


class Timings(BaseModel):
    parse: float
    search: float
    total: float


class SearchResponse(ParseResponse):
    results: list[Restaurant]
    timings_ms: Timings


class Anchor(BaseModel):
    name: str
    lat: float
    lon: float


# ---------- endpoints ----------


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


@router.post("", response_model=SearchResponse)
def search_restaurants(
    req: SearchRequest,
    parser: Annotated[Callable, Depends(get_parser)],
    encoder: Annotated[object, Depends(get_encoder)],
    db: Annotated[Callable, Depends(db_conn_provider)],
    data: Annotated[Callable, Depends(search_data_provider)],
) -> SearchResponse:
    start = time.perf_counter()
    parsed, engine_used = parser(req.text, req.engine)
    parse_ms = _ms(start)

    search_start = time.perf_counter()
    options = {"k": req.k, "include_closed": req.include_closed, "encoder": encoder}
    if req.backend == "sql":
        try:
            conn = db()
        except (psycopg.OperationalError, RuntimeError) as err:  # refused/timeout, or DATABASE_URL unset
            log.warning("search: can't connect to Postgres: %s", err)
            raise HTTPException(status_code=503, detail=DB_UNAVAILABLE) from err
        try:
            rows = search.search(conn, parsed, **options)
        except (psycopg.OperationalError, psycopg.InterfaceError) as err:  # connection dropped since cached
            log.warning("search: Postgres connection failed: %s", err)
            get_db_conn.cache_clear()
            raise HTTPException(status_code=503, detail=DB_UNAVAILABLE) from err
    else:
        rows = search.search_numpy(parsed, data=data(), **options)
    search_ms = _ms(search_start)

    return SearchResponse(
        parsed=ParsedQueryOut(**asdict(parsed)),
        engine_used=engine_used,
        results=[
            Restaurant(
                business_id=r["business_id"],
                name=r["name"],
                categories=r["categories"],
                price_level=r["price_level"],
                lat=r["latitude"],
                lon=r["longitude"],
                distance_km=r["distance_km"],
                similarity=r["similarity"],
            )
            for r in rows
        ],
        timings_ms=Timings(parse=parse_ms, search=search_ms, total=_ms(start)),
    )


@router.post("/parse", response_model=ParseResponse)
def parse_only(req: ParseRequest, parser: Annotated[Callable, Depends(get_parser)]) -> ParseResponse:
    parsed, engine_used = parser(req.text, req.engine)
    return ParseResponse(parsed=ParsedQueryOut(**asdict(parsed)), engine_used=engine_used)


@router.get("/anchors", response_model=list[Anchor])
def anchors() -> list[Anchor]:
    return [Anchor(name=name, lat=lat, lon=lon) for name, (lat, lon) in ANCHORS.items()]
