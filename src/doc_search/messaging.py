"""Всё про Kafka: топики, продюсер, консьюмер."""

from uuid import UUID

from confluent_kafka import Consumer, KafkaError, KafkaException, Message, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from doc_search.config import Settings
from doc_search.models import DocumentQueued


def ensure_topics(settings: Settings) -> None:
    """Создаёт топики, если их нет. 3 партиции — чтобы можно было запустить до 3 воркеров."""
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap})
    topics = [
        NewTopic(settings.kafka_topic, num_partitions=settings.kafka_partitions, replication_factor=1),
        NewTopic(settings.kafka_dlq_topic, num_partitions=1, replication_factor=1),
    ]
    for future in admin.create_topics(topics).values():
        try:
            future.result()
        except KafkaException as exc:
            if exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def make_producer(settings: Settings) -> Producer:
    return Producer({"bootstrap.servers": settings.kafka_bootstrap, "enable.idempotence": True})


def make_consumer(settings: Settings) -> Consumer:
    return Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap,
            "group.id": "doc-indexer",
            "auto.offset.reset": "earliest",
            # Offset коммитим сами и только ПОСЛЕ обработки — at-least-once.
            "enable.auto.commit": False,
            # Большой PDF может индексироваться минутами. Если poll() не вызывался дольше
            # этого времени, Kafka решит, что воркер умер, и отдаст партицию другому.
            "max.poll.interval.ms": 900_000,
        }
    )


def publish(producer: Producer, topic: str, key: bytes | None, value: bytes, timeout: float = 10.0) -> None:
    """Синхронная отправка: ждём подтверждения брокера, иначе — исключение."""
    errors: list[KafkaError] = []

    def on_delivery(err: KafkaError | None, _msg: Message) -> None:
        if err is not None:
            errors.append(err)

    producer.produce(topic, key=key, value=value, on_delivery=on_delivery)
    remaining = producer.flush(timeout)
    if remaining or errors:
        raise RuntimeError(f"Сообщение не доставлено в Kafka: {errors[0] if errors else 'таймаут'}")


def publish_document_queued(producer: Producer, settings: Settings, document_id: UUID) -> None:
    # key = document_id: все сообщения об одном документе попадают в одну партицию
    # и обрабатываются одним воркером строго по очереди.
    event = DocumentQueued(document_id=document_id)
    publish(producer, settings.kafka_topic, str(document_id).encode(), event.model_dump_json().encode())
