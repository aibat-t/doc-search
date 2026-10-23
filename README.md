# doc-search

Практика уроков **2.2.2 «Векторные базы»** и **2.2.3 «Индексация документов»**:
документы загружаются через API, индексируются воркером через Kafka в pgvector,
а поиск возвращает куски документов с источником, разделом и страницей.

```
POST /documents ──► storage/ + documents (status=queued) ──► Kafka: documents.queued
                                                               │ key = document_id
                                                               ▼
                                  doc-worker: парсинг → чанкинг → эмбеддинги
                                                               │ одна транзакция:
                                                               ▼ DELETE старых + INSERT новых
GET /search ◄──── HNSW (vector_cosine_ops) ◄──────────── chunks (pgvector)
```

## Запуск

```powershell
docker compose up -d          # Postgres + pgvector, Kafka
uv sync
uv run pytest

# в двух терминалах:
uv run doc-worker
uv run doc-api                # http://127.0.0.1:8000/docs
```

Загрузить примеры (или свои PDF/DOCX/HTML/MD/TXT):

```powershell
curl.exe -F "file=@samples/card-rules.md" http://127.0.0.1:8000/documents
curl.exe -F "file=@samples/tariffs.html" http://127.0.0.1:8000/documents
curl.exe "http://127.0.0.1:8000/documents"
curl.exe "http://127.0.0.1:8000/search?q=потерял карту что делать"
curl.exe "http://127.0.0.1:8000/search?q=сколько стоит перевод в другой банк&source=tariffs.html"
```

Замеры pgvector (урок 2.2.2):

```powershell
uv run doc-bench
uv run doc-bench --n 50000 --ef 10 40 100 400
```

## Структура

```
src/doc_search/
  config.py      настройки (DOC_*), см. .env.example
  schema.sql     таблицы documents и chunks, HNSW-индекс
  db.py          создание схемы, пул соединений
  parsing.py     PDF / HTML / DOCX / MD / TXT → блоки с разделом и страницей
  chunking.py    блоки → чанки с перекрытием и заголовком
  embedder.py    текст → вектор, счётчик токенов
  repository.py  все SQL-запросы
  messaging.py   Kafka: топики, продюсер, консьюмер
  indexer.py     индексация одного документа (идемпотентная)
  worker.py      цикл чтения Kafka
  api.py         FastAPI
  bench.py       перебор против HNSW: latency и recall
tests/           чанкинг и парсеры (без БД и Kafka)
samples/         учебные документы
```

## Что проверить руками

- [ ] Загрузить тот же файл ещё раз → ответ `unchanged`, воркер ничего не делает.
- [ ] Изменить файл и загрузить с тем же именем → чанки заменились, `chunk_count` обновился.
- [ ] Остановить воркер, загрузить файл, запустить воркер → документ проиндексируется из очереди.
- [ ] Загрузить скан PDF без текста → статус `failed`, причина в `error`, воркер жив, сообщение в `documents.dlq`.
- [ ] Поиск с `source=...` возвращает k результатов, а не меньше.

## Результаты замеров

| Вариант | среднее, мс | p95, мс | recall@10 |
|---|---|---|---|
| без индекса | | | 100% |
| HNSW, ef_search=10 | | | |
| HNSW, ef_search=40 | | | |
| HNSW, ef_search=100 | | | |
| HNSW, ef_search=200 | | | |

## Выводы

- …
