"""Сердце воркера: индексация одного документа. Kafka здесь не видна — её можно заменить чем угодно."""

import logging
from pathlib import Path
from uuid import UUID

from psycopg_pool import ConnectionPool

from doc_search import repository
from doc_search.chunking import chunk_blocks
from doc_search.config import Settings
from doc_search.embedder import Embedder
from doc_search.parsing import EmptyDocument, parse_file

log = logging.getLogger(__name__)


class Indexer:
    def __init__(self, pool: ConnectionPool, embedder: Embedder, settings: Settings) -> None:
        self._pool = pool
        self._embedder = embedder
        self._settings = settings

    def index(self, document_id: UUID) -> None:
        # Всё в одной транзакции: либо в базе полностью новая версия документа, либо старая.
        with self._pool.connection() as conn, conn.transaction():
            doc = repository.lock_document(conn, document_id)
            if doc is None:
                log.warning("Документ %s не найден — пропускаю", document_id)
                return
            if doc["indexed_hash"] == doc["content_hash"]:
                # Повторная доставка сообщения: эта версия уже в индексе.
                log.info("%s: эта версия уже проиндексирована — пропускаю", doc["source"])
                return

            parsed = parse_file(Path(doc["path"]), doc["source"])
            chunks = chunk_blocks(
                parsed.blocks,
                parsed.title,
                max_tokens=self._settings.chunk_max_tokens,
                overlap_tokens=self._settings.chunk_overlap_tokens,
                count_tokens=self._embedder.count_tokens,
            )
            if not chunks:
                raise EmptyDocument("После нарезки не осталось ни одного чанка")

            # Один вызов модели на все чанки — она сама разобьёт их на батчи.
            vectors = self._embedder.encode_documents([c.embed_text for c in chunks])

            repository.replace_chunks(conn, document_id, chunks, vectors)
            repository.mark_ready(conn, document_id, doc["content_hash"], len(chunks), self._embedder.name)

        log.info("%s: %d блоков → %d чанков", doc["source"], len(parsed.blocks), len(chunks))

    def mark_failed(self, document_id: UUID, error: str) -> None:
        with self._pool.connection() as conn:
            repository.mark_failed(conn, document_id, error)
