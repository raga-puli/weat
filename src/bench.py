"""Latency, throughput, and pgvector index benchmarks for search.

    python -m src.bench [--log "note"] [--runs 200] [--warmup 10] [--seconds 30]

Queries are the 40 in eval/queries.jsonl, parsed from the LLM cache (run
`python -m src.query_parser --eval` first; this never calls the LLM). Parsing and query embedding
are timed on their own; the search timings use precomputed query vectors, so they measure search only.

Query sets:
- unfiltered: ParsedQuery(soft=<raw query text>) for all 40, i.e. pure vector search.
- filtered: the parsed queries that have at least one hard constraint, as parsed.

Index comparisons use the unfiltered set (ivfflat applies WHERE filters after the index scan, so
filtered results are not comparable) and report recall@10 against the exact no-index results.
"""

import argparse
import json
import math
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from src import query_parser as qp
from src import search as s
from src.db import get_conn
from src.query_parser import ParsedQuery

K = 10
IVFFLAT_LISTS = (10, 50, 100)
INDEX_NAME = "bench_ivfflat"
RESULTS_PATH = Path("results/latency.md")
RESULTS_HEADER = (
    "| date | engine | index | params | filtered | p50_ms | p95_ms | qps | recall@10_vs_exact | build_s | note |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


class _NoLLM:
    def chat(self, **kwargs):
        raise RuntimeError("bench must not call the LLM; run `python -m src.query_parser --eval` to fill the cache")


def load_queries() -> tuple[list[str], list[ParsedQuery], str]:
    model = qp._model()
    texts = [json.loads(line)["query"] for line in qp.QUERIES_PATH.read_text(encoding="utf-8").splitlines() if line]
    missing = [t for t in texts if not qp._cache_path(model, t).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} queries are not in the cache for {model}, e.g. {missing[0]!r}")
    return texts, [qp.parse(t, client=_NoLLM(), model=model) for t in texts], model


def has_hard_constraint(p: ParsedQuery) -> bool:
    return bool(p.max_price_level or p.near or p.dietary or p.open_at or p.categories)


def latency(fn, items, runs: int, warmup: int) -> tuple[float, float]:
    for i in range(warmup):
        fn(items[i % len(items)])
    times = []
    for i in range(runs):
        start = time.perf_counter()
        fn(items[i % len(items)])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def throughput(fn, items, seconds: float) -> float:
    n = 0
    start = time.perf_counter()
    while (elapsed := time.perf_counter() - start) < seconds:
        fn(items[n % len(items)])
        n += 1
    return n / elapsed


def recall_at_k(got: list[list[str]], exact: list[list[str]]) -> float:
    scores = [len(set(g[:K]) & set(e[:K])) / len(e[:K]) for g, e in zip(got, exact) if e]
    return sum(scores) / len(scores) if scores else float("nan")


def _ids(results) -> list[str]:
    return [r["business_id"] for r in results]


def _uses_index(cur, parsed: ParsedQuery) -> bool:
    sql, params = s.build_query(parsed, K, s.embed(parsed.soft))
    cur.execute("EXPLAIN " + sql, params)
    return any(INDEX_NAME in row[0] for row in cur.fetchall())


def _row(engine, index, params, filtered, p50=None, p95=None, qps=None, recall=None, build_s=None, note=""):
    return {
        "engine": engine, "index": index, "params": params, "filtered": filtered, "p50_ms": p50,
        "p95_ms": p95, "qps": qps, "recall": recall, "build_s": build_s, "note": note,
    }


def run(runs: int, warmup: int, seconds: float) -> list[dict]:
    texts, parsed, model = load_queries()
    rows = []

    # Parsing (cache hits) and embedding, timed on their own.
    no_llm = _NoLLM()
    p50, p95 = latency(lambda t: qp.parse(t, client=no_llm, model=model), texts, runs, warmup)
    rows.append(_row("parse", "-", f"{model} {qp.PROMPT_VERSION}, cache hit", "-", p50, p95,
                     note="cached responses only; live LLM latency not measured"))
    real_embed = s.embed
    p50, p95 = latency(real_embed, texts, runs, warmup)
    rows.append(_row("embed", "-", s.MODEL_NAME, "-", p50, p95, note="query text -> vector"))

    unfiltered = [ParsedQuery(soft=t) for t in texts]
    filtered = [p for p in parsed if has_hard_constraint(p)]
    vectors = {q.soft: real_embed(q.soft) for q in unfiltered + filtered if q.soft}
    s.embed = vectors.__getitem__  # search timings below exclude embedding
    try:
        with get_conn() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")

            engines = {
                "numpy": lambda p: s.search_numpy(p, K),
                "postgres": lambda p: s.search(conn, p, K),
            }
            exact = [_ids(s.search(conn, p, K)) for p in unfiltered]
            for engine, fn in engines.items():
                for label, items in (("no", unfiltered), ("yes", filtered)):
                    print(f"timing {engine}, filtered={label} ({len(items)} queries) ...", flush=True)
                    p50, p95 = latency(fn, items, runs, warmup)
                    qps = throughput(fn, items, seconds)
                    recall = recall_at_k([_ids(fn(p)) for p in unfiltered], exact) if label == "no" else None
                    rows.append(_row(engine, "none", "exact", label, p50, p95, qps, recall,
                                     note=f"{len(items)} queries"))

            with conn.cursor() as cur:
                cur.execute("SHOW ivfflat.probes")
                default_probes = int(cur.fetchone()[0])
                for lists in IVFFLAT_LISTS:
                    print(f"building ivfflat lists={lists} ...", flush=True)
                    start = time.perf_counter()
                    cur.execute(
                        f"CREATE INDEX {INDEX_NAME} ON restaurants "
                        f"USING ivfflat (embedding vector_cosine_ops) WITH (lists = {int(lists)})"
                    )
                    cur.execute("ANALYZE restaurants")
                    build_s = time.perf_counter() - start
                    try:
                        for probes in sorted({default_probes, round(math.sqrt(lists))}):
                            cur.execute(f"SET ivfflat.probes = {int(probes)}")
                            fn = engines["postgres"]
                            p50, p95 = latency(fn, unfiltered, runs, warmup)
                            recall = recall_at_k([_ids(fn(p)) for p in unfiltered], exact)
                            used = _uses_index(cur, unfiltered[0])
                            label = "default" if probes == default_probes else "sqrt(lists)"
                            rows.append(_row("postgres", "ivfflat", f"lists={lists} probes={probes} ({label})", "no",
                                             p50, p95, None, recall, build_s,
                                             note="index used" if used else "planner did NOT use the index"))
                    finally:
                        cur.execute("RESET ivfflat.probes")
                        cur.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
    finally:
        s.embed = real_embed
    return rows


def _fmt(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_row(row: dict, note_suffix: str = "") -> str:
    note = "; ".join(x for x in (row["note"], note_suffix) if x)
    cells = [
        date.today().isoformat(), row["engine"], row["index"], row["params"], row["filtered"],
        _fmt(row["p50_ms"]), _fmt(row["p95_ms"]), _fmt(row["qps"], 0), _fmt(row["recall"], 3),
        _fmt(row["build_s"], 3), note,
    ]
    return "| " + " | ".join(cells) + " |"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help=f"note; also appends rows to {RESULTS_PATH}")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=30.0, help="throughput loop length per engine and set")
    args = parser.parse_args(argv)

    rows = run(args.runs, args.warmup, args.seconds)
    lines = [format_row(r, args.log or "") for r in rows]
    print()
    print(RESULTS_HEADER + "\n".join(lines))
    if args.log is not None:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not RESULTS_PATH.exists():
            RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
