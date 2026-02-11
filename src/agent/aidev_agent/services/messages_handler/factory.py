import os

from aidev_agent.utils.factory import SingletonFactory

from .base import BaseMessageQueueHandler
from .in_memory import InMemoryQueueMessageHandler

message_handler_factory: SingletonFactory[str, BaseMessageQueueHandler] = SingletonFactory(
    "message_handler", defaults=InMemoryQueueMessageHandler()
)

if os.getenv("RABBITMQ_HOST", ""):
    from .rabbitmq import RabbitMQMessageHandler

    message_handler_factory.replace_defaults(RabbitMQMessageHandler())
