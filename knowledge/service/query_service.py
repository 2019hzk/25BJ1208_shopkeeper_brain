import uuid, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED
from knowledge.utils.task_util import get_task_result


class QueryService:
    """

    """

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id():
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """
        运行查询流程的pineline
        Args:
            session_id:  会话id
            task_id:     任务id
            query:       查询问题
            is_stream:   是否是流式

        Returns:

        """
        # 1. 修改任务状态为正在执行
        update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)

        # 2. 构建查询初始化状态
        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream
        }

        # 3. 执行
        try:
            # 3.1 执行查询流程的pineline(调用的是CompiledStateGraph的invoke())
            query_app.invoke(query_init_state)
            # 3.2 更新整个任务状态为完成状态
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)

        # 4. 更新整个任务状态为异常状态
        except Exception as e:
            logger.error(f"运行查询流程出现异常:{str(e)}")
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def get_task_result(self, task_id: str):
        answer = get_task_result(task_id=task_id, key="answer")
        return answer
