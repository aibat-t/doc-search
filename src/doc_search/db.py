from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from doc_search.config import Settings

SCHEMA = Path(__file__).with_name("schema.sql")


def init_schema(settings: Settings) -> None:
    """Создаёт расширение и таблицы, если их нет. В настоящем проекте здесь были бы миграции Alembic."""
    sql = SCHEMA.read_text(encoding="utf-8").replace("{dim}", str(settings.embedding_dim))
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(sql)


def make_pool(settings: Settings) -> ConnectionPool:
    """Пул соединений (аналог HikariCP). register_vector учит psycopg отдавать numpy-массивы как vector.

    Вызывается после init_schema: register_vector требует, чтобы расширение уже существовало.
    """
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=5,
        kwargs={"row_factory": dict_row},
        configure=register_vector,
        open=False,
    )
    pool.open(wait=True)
    return pool
