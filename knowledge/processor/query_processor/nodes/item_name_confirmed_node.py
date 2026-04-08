import logging, re, json
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE


class _ItemNameExtractor:

    def extract_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """
        提取商品名
        Args:
            original_query: 用户原始查询
            history_context: 历史对话上下文

        Returns:

        """

        # 1. 定义LLM输出默认结果
        llm_result = {"item_names": [], "rewritten_query": original_query}

        # 2. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f"LLM客户端获取失败 原因:{str(e)}")
            return llm_result

        # 3. 获取商品名提取的提示词
        # 3.1 系统提示词
        item_name_system_prompt = "您是一位商品名提取专家，请从用户的问题以及历史对话中提取相关的商品名以及改写原始查询"
        # 3.2 用户提示词
        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context.strip() if history_context else "暂无历史上下文",
            query=original_query)

        # 4. 调用LLM
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=item_name_system_prompt),
                HumanMessage(content=item_name_user_prompt)
            ])
        except Exception as e:
            logger.error(f"LLM调用失败,原因：{str(e)}")
            return llm_result

        # 5. 获取LLM输出内容
        llm_response_content = llm_response.content

        # 6. 判断LLM的输出
        if not llm_response_content:
            return llm_result

        # 7. 清洗(判断输出内容的类型以及空格)和解析(反序列化)
        parsed_result: Dict[str, Any] = self._clean_and_parse(llm_response_content)

        # 8. 组装数据
        llm_result['item_names'] = parsed_result.get('item_names')
        llm_result['rewritten_query'] = parsed_result.get('rewritten_query') if parsed_result.get(
            'rewritten_query') else original_query

        # 9. 返回结果
        return llm_result

    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        """
        清洗以及解析LLM的结果
        Args:
            llm_response_content: llm的输出

        Returns:

        """
        # 1. 去除json代码块围栏标记```{}``` llm模型换了或者模型底层调用的API升级（防御性编程）
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 解析
        try:
            # 2.1 反序列化
            llm_content_obj: Dict[str, Any] = json.loads(content)

            # 2.2 获取item_names
            raw_item_names = llm_content_obj.get('item_names')

            # 2.3 判断类型
            if not isinstance(raw_item_names, list):
                item_names = []
            else:
                item_names = [item_name.strip() for item_name in raw_item_names if
                              isinstance(item_name, str) and item_name.strip()]

            # 2.4 获取rewritten_query
            raw_rewritten_query = llm_content_obj.get('rewritten_query')
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            # 2.5 返回
            return {
                "item_names": item_names,
                "rewritten_query": rewritten_query
            }
        except JSONDecodeError as e:
            logger.error(f"llm输出结果{llm_response_content} 反序列化失败 原因:{str(e)}")
            raise JSONDecodeError(msg=e.msg,
                                  doc=e.doc,
                                  pos=e.pos)


class _ItemNameAligner:
    pass


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        主要职责：
        1. 利用LLM从用户原始查询中提取商品名以及改写原始问查询（我喜欢你）
        1.1 如果LLM提取到了商品名，才进行第2步 去milvus对齐
        1.2 如果LLM没有提取到商品名，直接返回
        2. 根据Milvus中存储的商品名进行对齐（目的：检索更加的准确：三路检索都会利用该节点提取到的商品名，因此直接用LLM提取到商品名的话 下游三路检索在过滤的时候，过滤条件极其不准确。导致检索到的噪音很多 LLM最终输出的幻觉很高）
        最终不是要LLM的商品名 而是要Milvus中存储的商品名：因为milvus中没一个chunK都会关联milvus自己的商品名
        3. 决策（该走下去，还是回头）

        利用两个容器，产生三个分支：第一个分支去检索  第二个分支：给用户确认  第三个分支：抱歉
        1. confirmed:如果是精确的商品名--->给confirmed添加精确的商品名
        2. options:商品名不是精确，可是找到多个相似的---->给options中添加找到的多个不精确的商品名。

        state['answer']不要给，进行三路检索
        获取到三路检索结果
        把三路检索到的结果(RRF  RERANKER)给LLM
        LLM生成答案,在state['answer']
        state['answer']:就返回：
        1. 返回候选商品名【不精确】，给用户下一步确认使用
        2. 没有任何商品名，返回抱歉，没有找到您询问的关于任何商品的名字
        Args:
            state:
        Returns:
        """

        # 1. 获取用户原始问题
        original_query = state.get('original_query')

        # 2. 获取历史对话(mongodb)TODO
        history_context = ""

        # 3. 利用LLM进行商品名提取和查询重写
        llm_result: Dict[str, Any] = self._extractor.extract_item_name(original_query, history_context)

        return llm_result


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        "original_query": "RS-12数字万用表和华为擎云 L420x 分别如何测量电阻呢"
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)
