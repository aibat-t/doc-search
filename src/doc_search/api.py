"""HTTP API: загрузка документов и поиск.

    uv run doc-api   → http://127.0.0.1:8000/docs
"""

import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response, UploadFile

from doc_search import messaging, repository
from doc_search.config import Settings
from doc_search.db import init_schema, make_pool
from doc_search.embedder import Embedder
from doc_search.models import ChunkHit, DocumentOut, SearchResponse, UploadResult
from doc_search.parsing import SUPPORTED_SUFFIXES
from doc_search.reranker import Reranker
from doc_search.search import RerankerDisabled, Searcher, SearchMode

router = APIRouter()


@router.post("/documents", status_code=202)
def upload_document(request: Request, response: Response, file: UploadFile) -> UploadResult:
    """Сохраняет файл и ставит документ в очередь. Индексацию делает воркер — ответ сразу."""
    state = request.app.state
    source = Path(file.filename or "").name
    suffix = Path(source).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(415, f"Поддерживаются: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Пустой файл")

    content_hash = hashlib.sha256(data).hexdigest()
    # Имя файла в хранилище = хэш содержимого: одинаковые файлы хранятся один раз.
    path = state.settings.storage_dir / f"{content_hash}{suffix}"
    if not path.exists():
        path.write_bytes(data)

    with state.pool.connection() as conn:
        document_id, changed = repository.upsert_document(conn, source, str(path.resolve()), content_hash)

    if not changed:
        response.status_code = 200
        return UploadResult(id=document_id, status="unchanged", queued=False)

    # Публикуем после коммита. Если Kafka недоступна — документ останется в статусе queued,
    # его можно дотолкнуть через /reindex. Надёжное решение — паттерн transactional outbox.
    try:
        messaging.publish_document_queued(state.producer, state.settings, document_id)
    except RuntimeError as exc:
        raise HTTPException(503, f"{exc}. Повторите позже: POST /documents/{document_id}/reindex") from exc
    return UploadResult(id=document_id, status="queued", queued=True)


@router.post("/documents/{document_id}/reindex", status_code=202)
def reindex_document(request: Request, document_id: UUID) -> UploadResult:
    state = request.app.state
    with state.pool.connection() as conn:
        if not repository.requeue(conn, document_id):
            raise HTTPException(404, "Документ не найден")
    messaging.publish_document_queued(state.producer, state.settings, document_id)
    return UploadResult(id=document_id, status="queued", queued=True)


@router.get("/documents")
def list_documents(request: Request) -> list[DocumentOut]:
    with request.app.state.pool.connection() as conn:
        return [DocumentOut(**row) for row in repository.list_documents(conn)]


@router.get("/documents/{document_id}")
def get_document(request: Request, document_id: UUID) -> DocumentOut:
    with request.app.state.pool.connection() as conn:
        row = repository.get_document(conn, document_id)
    if row is None:
        raise HTTPException(404, "Документ не найден")
    return DocumentOut(**row)


@router.get("/search")
def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=1000, description="Вопрос пользователя")],
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    mode: Annotated[SearchMode, Query(description="vector | keyword | hybrid | hybrid_rerank")] = SearchMode.HYBRID,
    source: Annotated[str | None, Query(description="Искать только в этом файле")] = None,
) -> SearchResponse:
    """Поле score в ответе зависит от режима: косинус, ts_rank_cd, RRF или оценка reranker'а."""
    start = time.perf_counter()
    try:
        rows = request.app.state.searcher.search(q, mode, k, source)
    except RerankerDisabled as exc:
        raise HTTPException(400, str(exc)) from exc
    took_ms = (time.perf_counter() - start) * 1000
    return SearchResponse(query=q, mode=mode, took_ms=round(took_ms, 1), hits=[ChunkHit(**row) for row in rows])


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        init_schema(settings)
        messaging.ensure_topics(settings)
        app.state.settings = settings
        app.state.pool = make_pool(settings)
        app.state.producer = messaging.make_producer(settings)
        app.state.embedder = Embedder(settings.embedding_model, settings.query_prefix, settings.passage_prefix)
        reranker = Reranker(settings.rerank_model, settings.rerank_max_length) if settings.rerank_model else None
        app.state.searcher = Searcher(app.state.pool, app.state.embedder, settings, reranker)
        yield
        app.state.producer.flush(5)
        app.state.pool.close()

    app = FastAPI(title="Doc Search", version="0.1.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
