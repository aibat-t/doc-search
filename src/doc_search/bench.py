"""Урок 2.2.2: скорость и точность поиска в pgvector — без индекса и с HNSW.

    uv run doc-bench                         # 10 000 векторов, 100 запросов
    uv run doc-bench --n 50000 --ef 10 40 100 400

Работает со своей таблицей bench_vectors и не трогает документы.
"""

import argparse
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from doc_search.config import Settings

TOP_K_SQL = "SELECT id FROM bench_vectors ORDER BY embedding <=> %s LIMIT %s"


def make_dataset(n: int, n_queries: int, dim: int, clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Векторы, сгруппированные в «темы» (кластеры), — как у настоящих текстов.

    Чисто случайные векторы в 768 измерениях почти равноудалены друг от друга:
    у них нет «ближайших соседей», и любой ANN-индекс на них выглядит хуже, чем в жизни.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(clusters, dim))

    def sample(count: int) -> np.ndarray:
        points = centers[rng.integers(0, clusters, count)] + rng.normal(scale=0.5, size=(count, dim))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return points.astype(np.float32)

    return sample(n), sample(n_queries)


def load(conn: psycopg.Connection, vectors: np.ndarray) -> None:
    """COPY в бинарном формате — самый быстрый способ залить много строк (аналог batch insert)."""
    with conn.cursor() as cur, cur.copy("COPY bench_vectors (embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
        copy.set_types(["vector"])
        for vector in vectors:
            copy.write_row([vector])


def run_queries(conn: psycopg.Connection, queries: np.ndarray, k: int) -> tuple[list[list[int]], np.ndarray]:
    results, latencies = [], []
    for q in queries:
        start = time.perf_counter()
        rows = conn.execute(TOP_K_SQL, (q, k)).fetchall()
        latencies.append((time.perf_counter() - start) * 1000)
        results.append([row[0] for row in rows])
    return results, np.array(latencies)


def recall(approx: list[list[int]], exact: list[list[int]], k: int) -> float:
    """Какая доля настоящих top-k нашлась через индекс."""
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact, strict=True)]))


def explain(conn: psycopg.Connection, query: np.ndarray, k: int) -> str:
    literal = "[" + ",".join(f"{x:.6f}" for x in query) + "]"
    sql = f"EXPLAIN SELECT id FROM bench_vectors ORDER BY embedding <=> '{literal}'::vector LIMIT {int(k)}"
    return "\n".join(row[0] for row in conn.execute(sql).fetchall())


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="pgvector: перебор против HNSW")
    parser.add_argument("--n", type=int, default=10_000, help="сколько векторов в таблице")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--dim", type=int, default=settings.embedding_dim)
    parser.add_argument("--clusters", type=int, default=100)
    parser.add_argument("--ef", type=int, nargs="+", default=[10, 40, 100, 200], help="значения hnsw.ef_search")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data, queries = make_dataset(args.n, args.queries, args.dim, args.clusters, args.seed)

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS bench_vectors")
        conn.execute(f"CREATE TABLE bench_vectors (id bigserial PRIMARY KEY, embedding vector({args.dim}) NOT NULL)")

        start = time.perf_counter()
        load(conn, data)
        conn.execute("ANALYZE bench_vectors")
        print(f"Загружено {args.n:,} векторов × {args.dim} за {time.perf_counter() - start:.1f} с")

        results: list[tuple[str, np.ndarray, float]] = []

        # 1. Без индекса: Postgres перебирает все строки — это и есть «эталон» точности.
        run_queries(conn, queries[:5], args.k)          # прогрев: данные попадают в кэш
        exact_ids, latencies = run_queries(conn, queries, args.k)
        results.append(("без индекса (перебор)", latencies, 1.0))

        # 2. Строим HNSW.
        start = time.perf_counter()
        conn.execute("CREATE INDEX bench_hnsw ON bench_vectors USING hnsw (embedding vector_cosine_ops)")
        build = time.perf_counter() - start
        size = conn.execute("SELECT pg_size_pretty(pg_relation_size('bench_hnsw'))").fetchone()[0]
        print(f"HNSW построен за {build:.1f} с, размер индекса {size}")

        # 3. С индексом при разных ef_search: чем больше кандидатов, тем точнее и медленнее.
        for ef in args.ef:
            conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            run_queries(conn, queries[:5], args.k)
            ids, latencies = run_queries(conn, queries, args.k)
            results.append((f"HNSW, ef_search={ef}", latencies, recall(ids, exact_ids, args.k)))

        print(f"\n{'вариант':<24} {'среднее, мс':>12} {'p95, мс':>9} {f'recall@{args.k}':>10}")
        for name, lat, rec in results:
            print(f"{name:<24} {lat.mean():12.2f} {np.percentile(lat, 95):9.2f} {rec:10.1%}")

        print("\nПлан запроса с индексом (ищите строку 'Index Scan using bench_hnsw'):")
        print(explain(conn, queries[0], args.k))


if __name__ == "__main__":
    main()
