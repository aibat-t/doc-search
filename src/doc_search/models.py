from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentQueued(BaseModel):
    """Сообщение в Kafka: «документ нужно (пере)индексировать». Только id — всё остальное в БД."""

    document_id: UUID


class DocumentOut(BaseModel):
    id: UUID
    source: str
    status: str
    chunk_count: int
    error: str | None
    updated_at: datetime


class UploadResult(BaseModel):
    id: UUID
    status: str      # queued | unchanged
    queued: bool


class ChunkHit(BaseModel):
    chunk_id: int
    document_id: UUID
    source: str
    section: str | None
    page: int | None
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    took_ms: float
    hits: list[ChunkHit]
