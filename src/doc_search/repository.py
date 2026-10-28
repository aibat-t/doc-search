"""Все SQL-запросы проекта. Функции получают соединение снаружи — транзакцией управляет вызывающий."""

from typing import Any
from uuid import UUID

import numpy as np
from psycopg import Connection

from doc_search.chunking import Chunk

Row = dict[str, Any]


def upsert_document(conn: Connection, source: str, path: str, content_hash: str) -> tuple[UUID, bool]:
    """Регистрирует загрузку. Возвращает (id, нужно_ли_индексировать)."""
    inserted = conn.execute(
        """
        INSERT INTO documents (source, path, content_hash) VALUES (%s, %s, %s)
        ON CONFLICT (source) DO NOTHING
        RETURNING id
        """,
        (source, path, content_hash),
    ).fetchone()
    if inserted:
        return inserted["id"], True                   # новый документ

    row = conn.execute(
        "SELECT id, content_hash, status FROM documents WHERE source = %s FOR UPDATE", (source,)
    ).fetchone()
    if row["content_hash"] == content_hash and row["status"] != "failed":
        return row["id"], False                       # тот же файл ещё раз — ничего не делаем

    conn.execute(
        """
        UPDATE documents
        SET path = %s, content_hash = %s, status = 'queued', error = NULL, updated_at = now()
        WHERE id = %s
        """,
        (path, content_hash, row["id"]),
    )
    return row["id"], True                            # новая версия или повтор после ошибки


def requeue(conn: Connection, document_id: UUID) -> bool:
    """Принудительная переиндексация: сбрасываем indexed_hash, чтобы воркер не пропустил документ."""
    row = conn.execute(
        """
        UPDATE documents SET status = 'queued', indexed_hash = NULL, error = NULL, updated_at = now()
        WHERE id = %s RETURNING id
        """,
        (document_id,),
    ).fetchone()
    return row is not None


def lock_document(conn: Connection, document_id: UUID) -> Row | None:
    """FOR UPDATE: второй воркер с тем же документом будет ждать, пока первый закончит."""
    return conn.execute(
        "SELECT id, source, path, content_hash, indexed_hash FROM documents WHERE id = %s FOR UPDATE",
        (document_id,),
    ).fetchone()


def replace_chunks(conn: Connection, document_id: UUID, chunks: list[Chunk], vectors: np.ndarray) -> None:
    """Удалить все старые чанки и вставить новые — в той же транзакции, что и mark_ready."""
    conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, header, text, section, page, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (document_id, c.index, c.header, c.text, c.section or None, c.page, vector)
                for c, vector in zip(chunks, vectors, strict=True)
            ],
        )


def mark_ready(conn: Connection, document_id: UUID, content_hash: str, chunk_count: int, model: str) -> None:
    conn.execute(
        """
        UPDATE documents
        SET status = 'ready', indexed_hash = %s, chunk_count = %s, embedding_model = %s,
            error = NULL, updated_at = now()
        WHERE id = %s
        """,
        (content_hash, chunk_count, model, document_id),
    )


def mark_failed(conn: Connection, document_id: UUID, error: str) -> None:
    conn.execute(
        "UPDATE documents SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
        (error[:1000], document_id),
    )


_DOCUMENT_COLUMNS = "id, source, status, chunk_count, error, updated_at"


def list_documents(conn: Connection) -> list[Row]:
    return conn.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY updated_at DESC").fetchall()


def get_document(conn: Connection, document_id: UUID) -> Row | None:
    return conn.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE id = %s", (document_id,)).fetchone()


def search_chunks(
    conn: Connection, query_vector: np.ndarray, k: int, ef_search: int, source: str | None = None
) -> list[Row]:
    """Семантический поиск. Должен выполняться внутри транзакции — из-за SET LOCAL."""
    # SET не принимает параметры запроса, поэтому подставляем число через int() — это безопасно.
    conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
    where = ""
    params: dict[str, Any] = {"q": query_vector, "k": k}
    if source:
        # Без этого HNSW найдёт ef_search кандидатов, отфильтрует, и строк может остаться меньше k.
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        where = "WHERE d.source = %(source)s"
        params["source"] = source

    rows = conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.document_id, d.source, c.header, c.section, c.page, c.text,
               1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
        """,
        params,
    ).fetchall()
    # relaxed_order может слегка нарушить порядок — досортируем сами.
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def keyword_search(conn: Connection, query: str, k: int, source: str | None = None) -> list[Row]:
    """Полнотекстовый поиск по колонке tsv (GIN-индекс). Слова запроса соединяем через ИЛИ.

    plainto_tsquery('russian', ...) делает стемминг и выкидывает стоп-слова, но соединяет слова через И:
    'скольк' & 'сто' & 'перевод' & 'банк'. На естественный вопрос это часто даёт ноль строк —
    все слова должны встретиться в одном чанке. Меняем & на | и приводим обратно к tsquery
    (без повторного стемминга). Чанки, где совпало больше слов, всё равно окажутся выше в ранжировании.
    """
    where = ""
    params: dict[str, Any] = {"query": query, "k": k}
    if source:
        where = "AND d.source = %(source)s"
        params["source"] = source
    return conn.execute(
        f"""
        SELECT c.id AS chunk_id, c.document_id, d.source, c.header, c.section, c.page, c.text,
               ts_rank_cd(c.tsv, q, 1) AS score       -- 1 = поправка на длину чанка
        FROM chunks c
        JOIN documents d ON d.id = c.document_id,
             CAST(replace(plainto_tsquery('russian', %(query)s)::text, ' & ', ' | ') AS tsquery) AS q
        WHERE c.tsv @@ q {where}
        ORDER BY score DESC
        LIMIT %(k)s
        """,
        params,
    ).fetchall()
