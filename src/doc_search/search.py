"""Четыре режима поиска за одним методом. Им пользуются и API, и doc-eval.

    vector         — только векторы (pgvector, HNSW)
    keyword        — только слова (tsvector, GIN)
    hybrid         — оба поиска по top-N кандидатов → слияние RRF
    hybrid_rerank  — hybrid, а потом cross-encoder пересортировывает кандидатов
"""

from enum import StrEnum
from typing import Any

from psycopg_pool import ConnectionPool

from doc_search import repository
from doc_search.config import Settings
from doc_search.embedder import Embedder
from doc_search.ranking import rrf
from doc_search.reranker import Reranker

Row = dict[str, Any]


class SearchMode(StrEnum):
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class RerankerDisabled(RuntimeError):
    pass


class Searcher:
    def __init__(
        self, pool: ConnectionPool, embedder: Embedder, settings: Settings, reranker: Reranker | None = None
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._settings = settings
        self._reranker = reranker

    def search(self, query: str, mode: SearchMode, k: int = 5, source: str | None = None) -> list[Row]:
        """Возвращает top-k чанков. Смысл поля score зависит от режима — сравнивать его между режимами нельзя."""
        if mode is SearchMode.VECTOR:
            return self._vector(query, k, source)
        if mode is SearchMode.KEYWORD:
            with self._pool.connection() as conn:
                return repository.keyword_search(conn, query, k, source)

        candidates = self._hybrid(query, self._settings.search_candidates, source)
        if mode is SearchMode.HYBRID:
            return candidates[:k]
        return self._rerank(query, candidates, k)

    def _vector(self, query: str, k: int, source: str | None) -> list[Row]:
        query_vector = self._embedder.encode_queries([query])[0]
        # Подвох HNSW: индекс возвращает не больше ef_search строк. При ef_search=40 запрос
        # «дай 50 кандидатов» молча вернёт 40. Поэтому ef_search не меньше k.
        ef_search = max(self._settings.hnsw_ef_search, k)
        with self._pool.connection() as conn, conn.transaction():
            return repository.search_chunks(conn, query_vector, k, ef_search, source)

    def _hybrid(self, query: str, n: int, source: str | None) -> list[Row]:
        by_vector = self._vector(query, n, source)
        with self._pool.connection() as conn:
            by_keyword = repository.keyword_search(conn, query, n, source)

        # Каждый поиск выполняется отдельным запросом и использует свой индекс (HNSW и GIN),
        # а слияние — десяток строк на Python. Одним SQL это сложнее и планировщику, и читателю.
        rows = {row["chunk_id"]: row for row in by_keyword} | {row["chunk_id"]: row for row in by_vector}
        fused = rrf(
            [row["chunk_id"] for row in by_vector],
            [row["chunk_id"] for row in by_keyword],
            k=self._settings.rrf_k,
        )
        return [{**rows[chunk_id], "score": score} for chunk_id, score in fused[:n]]

    def _rerank(self, query: str, candidates: list[Row], k: int) -> list[Row]:
        if self._reranker is None:
            raise RerankerDisabled("Reranker выключен: задайте DOC_RERANK_MODEL")
        # Reranker видит то же, что эмбеддер: заголовок + текст. Без заголовка он не узнает,
        # что кусок «Абонент вправе сохранить номер...» — это статья 36-1.
        scores = self._reranker.score(query, [f"{row['header']}\n{row['text']}" for row in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [{**row, "score": score} for row, score in ranked[:k]]
