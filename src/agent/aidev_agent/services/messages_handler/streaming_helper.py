import threading
import time
import uuid
from logging import getLogger
from typing import Any, Generator

from .base import (
    CANCELLED_CHUNK,
    EOD_CHUNK,
    HEARTBEAT_CHUNK,
    HEARTBEAT_INTERVAL,
    HEARTBEAT_TIMEOUT,
    BaseMessageQueueHandler,
)
from .factory import message_handler_factory

logger = getLogger(__name__)


class GeneratorStreamingHelper:
    """生成器流式处理辅助类

    使用死信队列机制支持断点续传的流式处理：
    - 通过 has_pending_messages() 判断是否需要创建新的生产者
    - 消费者读取消息后，消息从主队列移动到死信队列
    - 消费者断开后重连时，先调用 restore_messages() 恢复消息，再继续消费
    - 读到 EOD_CHUNK 时调用 mark_completed() 清理所有队列

    心跳机制：
    - 生产者在数据产生间隔较长时，定期发送心跳消息
    - 消费者检测心跳超时，如果超过 HEARTBEAT_TIMEOUT 未收到任何消息，则认为生产者异常

    工作流程：
    1. 客户端首次请求时，队列为空，启动生产者线程生产数据
    2. 消费者从主队列获取消息，消息被移动到死信队列
    3. 如果客户端断开连接，生产者继续运行直到完成
    4. 客户端重连时，检查队列中是否有数据：
       - 如果有数据，先调用 restore_messages() 恢复消息，再消费
       - 如果没有数据，启动新的生产者
    5. 读到 EOD_CHUNK 时，调用 mark_completed() 清理队列
    """

    def __init__(self, message_handler: BaseMessageQueueHandler | None = None, thread_id: str | None = None) -> None:
        self.message_handler = message_handler if message_handler else message_handler_factory.get()
        self.thread_id = thread_id or uuid.uuid4().hex

    def stream(self, generator: Generator[Any, None, None]) -> Generator[Any, None, None]:
        """使用队列处理器缓存流式请求

        Args:
            generator: 数据生成器

        Yields:
            生成器产生的数据

        Raises:
            RuntimeError: 当心跳超时时抛出，表示生产者可能已异常结束
        """
        # 检查队列中是否有未消费的数据
        has_pending = self.message_handler.has_pending_messages(self.thread_id)

        producer_thread = None
        # 记录最后一次收到消息的时间（用于心跳超时检测）
        last_message_time = time.time()
        # 是否启用心跳检测（仅在启动新生产者时启用）
        enable_heartbeat_check = False

        if not has_pending:
            # 队列为空，需要启动新的生产者
            # 先清空队列确保干净状态
            self.message_handler.clear(self.thread_id)

            # 启动生产者线程
            producer_thread = threading.Thread(target=self._producer, args=(generator,), daemon=True)
            producer_thread.start()
            logger.info(f"Started producer for thread_id={self.thread_id}")
            enable_heartbeat_check = True
        else:
            # 队列中有数据，不启动新的生产者
            # 先将死信队列的消息恢复到主队列，从头消费
            restored = self.message_handler.restore_messages(self.thread_id)
            logger.info(
                f"Pending messages exist for thread_id={self.thread_id}, "
                f"restored {restored} messages from DLQ, consuming from start"
            )

        # 消费者：从队列中获取消息
        try:
            while True:
                try:
                    # 从主队列获取消息（消息会被移动到死信队列）
                    messages = self.message_handler.get(self.thread_id, timeout=0.5)

                    # 收到消息，更新最后消息时间
                    if messages:
                        last_message_time = time.time()

                    # 处理获取到的消息
                    for item in messages:
                        if item == HEARTBEAT_CHUNK:
                            # 跳过心跳消息，不向消费者返回
                            logger.debug(f"Received heartbeat for thread_id={self.thread_id}")
                            continue
                        if item == EOD_CHUNK:
                            # 读到结束标记，调用 mark_completed 清理所有队列
                            self.message_handler.mark_completed(self.thread_id)
                            logger.info(f"Stream completed for thread_id={self.thread_id}")
                            return
                        if item == CANCELLED_CHUNK:
                            # 主动取消，同样清理队列并结束
                            self.message_handler.mark_completed(self.thread_id)
                            logger.info(f"Stream cancelled for thread_id={self.thread_id}")
                            return
                        yield item
                except TimeoutError:
                    # 超时，检查心跳是否超时
                    if enable_heartbeat_check and (time.time() - last_message_time > HEARTBEAT_TIMEOUT):
                        logger.error(f"心跳超时 thread_id={self.thread_id}，超过 {HEARTBEAT_TIMEOUT}s 未收到任何消息")
                        raise RuntimeError(
                            f"生产者心跳超时：超过 {HEARTBEAT_TIMEOUT}s 未收到任何消息，生产者可能已崩溃"
                        )
                    continue
        except GeneratorExit:
            # 客户端断开连接，不清理队列，消息已在死信队列中保留
            logger.info(f"Consumer disconnected for thread_id={self.thread_id}, messages preserved in DLQ")
            raise
        finally:
            # 等待生产者线程结束（如果是本次启动的）
            if producer_thread is not None:
                try:
                    producer_thread.join(timeout=2.0)
                except Exception as e:
                    logger.exception(f"Error joining producer thread for thread_id={self.thread_id}: {e}")

    def _producer(self, generator: Generator[Any, None, None]) -> None:
        """生产者线程：将生成器产生的消息推送到队列

        即使消费者断开连接，生产者也会继续运行直到完成。
        会定期发送心跳消息，让消费者知道生产者仍然存活。
        """

        last_heartbeat_time = time.time()
        cancelled = False
        try:
            for chunk in generator:
                if self.message_handler.is_cancel_requested(self.thread_id):
                    cancelled = True
                    break
                self.message_handler.put(self.thread_id, chunk)
                logger.debug(f"Produced chunk for thread_id={self.thread_id}")
                # 检查是否需要发送心跳
                current_time = time.time()
                if current_time - last_heartbeat_time >= HEARTBEAT_INTERVAL:
                    self.message_handler.put(self.thread_id, HEARTBEAT_CHUNK)
                    last_heartbeat_time = current_time
                    logger.debug(f"Sent heartbeat for thread_id={self.thread_id}")
        except GeneratorExit:
            logger.info(f"Generator closed for thread_id={self.thread_id}")
        except Exception as e:
            logger.debug(f"Sent error chunk for thread_id={self.thread_id}: {e}")
        finally:
            # 生产者完成：主动取消发 CANCELLED_CHUNK，否则发 EOD_CHUNK
            end_chunk = CANCELLED_CHUNK if cancelled else EOD_CHUNK
            self.message_handler.put(self.thread_id, end_chunk)
            self.message_handler.flush(self.thread_id)
            logger.debug(f"Producer finished, sent {end_chunk} for thread_id={self.thread_id}")
