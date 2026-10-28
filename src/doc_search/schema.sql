CREATE EXTENSION IF NOT EXISTS vector;

-- Один документ = один загруженный файл. source (имя файла) — его «естественный ключ»:
-- повторная загрузка файла с тем же именем — это новая версия того же документа.
CREATE TABLE IF NOT EXISTS documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source          text NOT NULL UNIQUE,
    path            text NOT NULL,
    content_hash    text NOT NULL,         -- хэш последней загруженной версии
    indexed_hash    text,                  -- хэш версии, которая лежит в chunks
    status          text NOT NULL DEFAULT 'queued',   -- queued | ready | failed
    error           text,
    chunk_count     int  NOT NULL DEFAULT 0,
    embedding_model text,
    updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  int  NOT NULL,
    header       text NOT NULL,    -- «Название документа > Раздел»
    text         text NOT NULL,    -- сам кусок, без заголовка — его показываем пользователю
    section      text,
    page         int,
    embedding    vector({dim}) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- vector_cosine_ops ↔ оператор <=> в запросах. Не совпадут — индекс молча не используется.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_id ON chunks (document_id);

-- Урок 2.2.4: полнотекстовый поиск (BM25-подобный) рядом с векторным.
-- Генерируемая колонка: Postgres сам пересчитывает её при каждом INSERT/UPDATE чанка.
-- В текст включаем header («Документ > Глава > Статья 36-1. ...»): так запрос «статья 36-1» находит статью.
-- 'russian' — стемминг: «карты», «картой», «карту» → «карт». Для казахских текстов он не подходит —
-- там нужна 'simple' (без стемминга) или отдельная колонка под каждый язык.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', header || ' ' || text)) STORED;

-- GIN — инвертированный индекс «слово → список чанков», как в Lucene/Elasticsearch.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
