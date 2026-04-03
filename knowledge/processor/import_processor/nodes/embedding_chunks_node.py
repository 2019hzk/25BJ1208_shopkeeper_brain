from typing import List, Dict, Any
from pathlib import Path
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, EmbeddingError
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验state的chunks
        self.log_step("step1", "校验chunks的数据结构")
        validated_chunks = self._validate_state(state)

        # 2. 获取嵌入模型
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)

        # 3. 批量嵌入
        # 3.1 获取批量阈值
        batch_size = self.config.embedding_batch_size

        # 3.2 获取chunks的总数
        total = len(validated_chunks)

        # 3.3 遍历
        final_chunks = []
        for index in range(0, total, batch_size):
            # 获取当前这一批
            bath_chunks = validated_chunks[index:index + batch_size]
            # 获取当前这个一批的最后一个编号
            batch_end = index + len(bath_chunks)
            self.logger.info(f"嵌入批次 [{index + 1}-{batch_end}] / {total}")
            current_chunks = self._embed_chunks(bath_chunks, embed_model)
            final_chunks.extend(current_chunks)

        # 4. 更新state的chunks
        state['chunks'] = final_chunks
        # 5. 返回
        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        # 1. 获取chunks
        chunks = state.get('chunks')

        # 2. 校验chunks类型
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 3. 遍历检验每一个chunk的类型
        for index, chunk in enumerate(chunks):

            # 校验单个chunk
            if not isinstance(chunk, dict):
                raise ValidationError(
                    message=f"[chunk_{index + 1}] 类型和期望的类型不匹配，实际的类型{type(chunk).__name__}",
                    node_name=self.name)

        # 4. 返回chunks
        return chunks

    def _embed_chunks(self, bath_chunks: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction) -> List[
        Dict[str, Any]]:
        """
        批量嵌入chunks
        Args:
            bath_chunks: 批量chunks
            embed_model: 嵌入模型

        Returns:

        """
        # 1. 获取要嵌入的内容
        embedding_documents = [f"{chunk.get('item_name', '')}\n{chunk.get('content', '')}" for chunk in bath_chunks]

        # 2. 嵌入chunks的真正内容
        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}", node_name=self.name)

        if not embed_vector:
            raise EmbeddingError(message="嵌入结果不存在")

        # 3. 遍历这一批的每一个chunk
        sparse_csr = embed_vector.get('sparse')
        for i, chunk in enumerate(bath_chunks):
            chunk['dense_vector'] = embed_vector.get('dense')[i].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(sparse_csr, i)

        return bath_chunks

    def _extract_sparse_vector(self, sparse_csr, index: int):
        """
        从稀疏矩阵中提取当前chunk对象的稀疏向量
        Args:
            sparse_csr:
            index:

        Returns:

        """
        # 3.1 从行索引中获取当前chunk的起始索引和结束索引
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        # 3.2 获取token_id
        token_id = sparse_csr.indices[start_index:end_index].tolist()
        # 3.3 获取weight
        weight = sparse_csr.data[start_index:end_index].tolist()

        # 3.4 返回单个chunk的稀疏向量值
        return dict(zip(token_id, weight))


if __name__ == '__main__':
    import json

    setup_logging()

    base_dir = Path(
        r"D:\develop\develop\workspace\pycharm\BJ251208\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
    )
    input_path = base_dir / "chunks_item_name.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks')})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")
