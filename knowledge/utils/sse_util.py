import json
import queue
from typing import Dict, Any, Optional, AsyncGenerator



class SSEEvent:
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案


# 全局 SSE 任务队列存储
# Key: task_id, Value: queue.Queue
_task_stream: Dict[str, queue.Queue] = {}


def get_sse_queue(task_id: str) -> Optional[queue.Queue]:
    """获取指定任务的队列"""
    return _task_stream.get(task_id)


def create_sse_queue(task_id: str) -> queue.Queue:
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _task_stream[task_id] = q
    return q

def remove_sse_queue(task_id: str):
    """移除指定任务的队列
    不存在 key 默认返回 None
    """
    _task_stream.pop(task_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_sse_event (task_id: str, event: str, data: Dict[str, Any]):
    """
    通过 task_id 推送事件到 SSE 队列
    """
    # 1. 获取 SSE 队列
    stream_queue = get_sse_queue(task_id)

    # 2. 队列存在
    if stream_queue:
        # 3. 将事件推送到队列
        stream_queue.put({"event": event, "data": data})



