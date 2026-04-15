"""第三步：端到端验证

使用 search 找到的最佳阈值，走完整的 ItemNameConfirmedNode.process() 流程。
需要连接 LLM + Milvus + 嵌入模型。
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from knowledge.test.threshold.threshold_config import ThresholdConfig, logger


def verify(test_cases: List[Dict[str, Any]], config: ThresholdConfig = None) -> None:
    """端到端验证"""
    config = config or ThresholdConfig()

    result_path = Path(config.result_path)
    if not result_path.exists():
        logger.error("找不到搜索结果文件，请先运行 search")
        return

    with open(result_path, "r", encoding="utf-8") as f:
        search_result = json.load(f)

    best = search_result["best_params"]
    logger.info("=" * 60)
    logger.info("端到端验证（含 LLM 提取 + 向量匹配）")
    logger.info(f"使用阈值: confirm={best['confirm']}, options={best['options']}, gap={best['gap']}")
    logger.info("=" * 60)

    # 将最佳阈值写入全局配置，这样 ItemNameConfirmedNode 内部的 _ItemNameAligner
    # 通过 get_config() 拿到的就是 best 阈值，而不是默认值
    from knowledge.processor.query_processor.base import get_config
    query_config = get_config()
    query_config.item_name_high_confidence = best["confirm"]
    query_config.item_name_mid_confidence = best["options"]
    query_config.item_name_score_gap = best["gap"]

    # 使用真实节点（此时节点内部读取的配置已经是 best 阈值）
    from knowledge.processor.query_processor.nodes.item_name_confirmed_node import ItemNameConfirmedNode

    node = ItemNameConfirmedNode()

    for i, case in enumerate(test_cases, 1):
        query = case["query"]
        logger.info(f"\n--- 用例 {i}: {case['description']} ---")
        logger.info(f"  问题: {query}")
        logger.info(f"  期望 confirmed: {case['expected_confirmed']}")

        state = {
            "original_query": query,
            "session_id": f"test_threshold_{i}",
        }

        try:
            result = node.process(state)
            confirmed = result.get("item_names", [])
            answer = result.get("answer", "")

            logger.info(f"  实际 confirmed: {confirmed}")
            if answer:
                logger.info(f"  拦截回复: {answer[:80]}...")

            # 判断结果
            expected = case["expected_confirmed"]
            if not expected and not confirmed:
                logger.info("  结果: ✓ (两者都为空)")
            elif set(expected) <= set(confirmed or []):
                logger.info("  结果: ✓")
            else:
                logger.info("  结果: ✕ (不匹配)")

        except Exception as e:
            logger.error(f"  执行失败: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info("验证完成")


if __name__ == "__main__":
    from knowledge.test.threshold.test_case import TEST_CASES

    verify(TEST_CASES)