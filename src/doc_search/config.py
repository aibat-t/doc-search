from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки API, воркера и бенчмарка. Переменные окружения с префиксом DOC_."""

    model_config = SettingsConfigDict(env_prefix="DOC_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/docs"

    kafka_bootstrap: str = "localhost:9092"
    kafka_topic: str = "documents.queued"
    kafka_dlq_topic: str = "documents.dlq"
    kafka_partitions: int = 3

    storage_dir: Path = Path("storage")

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    # Теперь задача асимметричная: короткий вопрос ищет длинный кусок текста.
    # Поэтому у e5 разные префиксы: "query: " для вопроса, "passage: " для чанка.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    chunk_max_tokens: int = 350      # + заголовок и префикс — укладываемся в лимит e5 (512)
    chunk_overlap_tokens: int = 50   # ≈ 15% перекрытия

    hnsw_ef_search: int = 40

    # Урок 2.2.4: гибридный поиск и reranking.
    search_candidates: int = 50      # сколько кандидатов берём из каждого поиска перед слиянием/reranking
    rrf_k: int = 60                  # константа из статьи про RRF, менять почти никогда не нужно
    # Мультиязычный cross-encoder (~570M параметров, на CPU 50 пар — секунды).
    # Быстрее, но слабее: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1. Пустая строка — reranker выключен.
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 512
