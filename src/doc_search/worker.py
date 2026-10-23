"""Воркер: читает Kafka и индексирует документы.

    uv run doc-worker
"""

import logging

from confluent_kafka import Message, Producer
from pydantic import ValidationError

from doc_search import messaging
from doc_search.config import Settings
from doc_search.db import init_schema, make_pool
from doc_search.embedder import Embedder
from doc_search.indexer import Indexer
from doc_search.models import DocumentQueued

log = logging.getLogger("doc_search.worker")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    init_schema(settings)
    messaging.ensure_topics(settings)

    log.info("Загружаю модель %s ...", settings.embedding_model)
    embedder = Embedder(settings.embedding_model, settings.query_prefix, settings.passage_prefix)
    pool = make_pool(settings)
    indexer = Indexer(pool, embedder, settings)
    producer = messaging.make_producer(settings)
    consumer = messaging.make_consumer(settings)
    consumer.subscribe([settings.kafka_topic])
    log.info("Слушаю топик %s (Ctrl+C — остановить)", settings.kafka_topic)

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("Kafka: %s", msg.error())
                continue
            handle(msg, indexer, producer, settings)
            # Коммит ПОСЛЕ обработки. Упадём между handle и commit — сообщение придёт ещё раз,
            # поэтому обработка обязана быть идемпотентной (см. Indexer.index).
            consumer.commit(message=msg, asynchronous=False)
    except KeyboardInterrupt:
        log.info("Остановка")
    finally:
        consumer.close()
        producer.flush(5)
        pool.close()


def handle(msg: Message, indexer: Indexer, producer: Producer, settings: Settings) -> None:
    try:
        event = DocumentQueued.model_validate_json(msg.value())
    except ValidationError:
        log.error("Некорректное сообщение, отправляю в DLQ: %r", msg.value())
        messaging.publish(producer, settings.kafka_dlq_topic, msg.key(), msg.value())
        return

    try:
        indexer.index(event.document_id)
    except Exception as exc:
        # Битый PDF не должен ронять воркер: помечаем документ и откладываем сообщение в DLQ.
        # Упрощение: так же поступаем и с временными ошибками (БД недоступна) —
        # в проде для них нужен повтор с паузой.
        log.exception("Не удалось проиндексировать %s", event.document_id)
        indexer.mark_failed(event.document_id, f"{type(exc).__name__}: {exc}")
        messaging.publish(producer, settings.kafka_dlq_topic, msg.key(), msg.value())


if __name__ == "__main__":
    main()
