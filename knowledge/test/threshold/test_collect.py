"""第一步：采集分数数据

对每个测试用例调用向量数据库获取真实分数并保存到文件。
需要连接 Milvus + 嵌入模型。
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query

from knowledge.test.threshold.threshold_config import ThresholdConfig, logger


def collect(test_cases: List[Dict[str, Any]], config: ThresholdConfig = None) -> None:
    """采集分数数据"""
    config = config or ThresholdConfig()

    logger.info("=" * 60)
    logger.info("开始采集分数数据")
    logger.info("=" * 60)

    # 初始化客户端
    milvus_client = StorageClients.get_milvus_client()
    embedding_model = AIClients.get_bge_m3_client()
    if not milvus_client or not embedding_model:
        logger.error("Milvus 或嵌入模型初始化失败")
        return

    collected_data = []

    for i, case in enumerate(test_cases, 1):
        logger.info(f"\n--- 用例 {i}/{len(test_cases)}: {case['description']} ---")
        logger.info(f"  问题: {case['query']}")
        logger.info(f"  LLM提取: {case['llm_extract']}")

        item_names = case["llm_extract"]
        if not item_names:
            logger.info("  (无提取结果，跳过向量检索)")
            collected_data.append({**case, "search_results": []})
            continue

        # 嵌入
        embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        search_results = []
        for idx, extract_name in enumerate(item_names):
            # 创建混合检索请求
            requests = create_hybrid_search_requests(
                dense_vector=embedding_result['dense'][idx],
                sparse_vector=embedding_result['sparse'][idx],
            )

            # 执行检索
            result = execute_hybrid_search_query(
                milvus_client,
                collection_name=config.collection_name,
                search_requests=requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )

            matches = [
                {"item_name": h["entity"]["item_name"], "score": round(h["distance"], 6)}
                for h in (result[0] if result else [])
            ]

            search_results.append({
                "extracted_name": extract_name,
                "matches": matches
            })

            # 打印分数分布
            logger.info(f"  [{extract_name}] 匹配结果:")
            for m in matches[:5]:
                logger.info(f"    {m['item_name']:30s}  score={m['score']:.4f}")

        collected_data.append({**case, "search_results": search_results})

    # 保存到文件
    output_path = Path(config.data_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n分数数据已保存到: {output_path.absolute()}")
    logger.info(f"共采集 {len(collected_data)} 条用例")

    # 打印分数分布概览
    _print_score_summary(collected_data)


def _print_score_summary(collected_data: List[Dict]) -> None:
    """打印所有分数的分布概览"""
    all_scores = []
    for case in collected_data:
        for sr in case.get("search_results", []):
            for m in sr.get("matches", []):
                all_scores.append(m["score"])

    if not all_scores:
        return

    all_scores.sort(reverse=True)

    logger.info("\n" + "=" * 60)
    logger.info("分数分布概览")
    logger.info("=" * 60)

    ranges = [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6), (0.0, 0.5)]
    for low, high in ranges:
        count = sum(1 for s in all_scores if low <= s < high)
        bar = "█" * count
        logger.info(f"  [{low:.1f}, {high:.1f})  {count:3d} 个  {bar}")

    logger.info(f"\n  总数: {len(all_scores)}, 最高: {max(all_scores):.4f}, 最低: {min(all_scores):.4f}")


if __name__ == "__main__":
    from knowledge.test.threshold.test_case import TEST_CASES

    collect(TEST_CASES)