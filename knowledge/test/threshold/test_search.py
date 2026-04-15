"""第二步：搜索最佳阈值

读取 collect 采集的分数数据，遍历所有阈值组合，找到最佳参数。
纯本地计算，不需要连接外部服务。
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

from knowledge.test.threshold.threshold_config import (
    ThresholdConfig, logger,
    simulate_align, simulate_filter, check_result
)


def search(config: ThresholdConfig = None) -> Dict[str, Any]:
    """搜索最佳阈值组合"""

    # 1. 加载配置（没传就用默认配置）
    config = config or ThresholdConfig()

    # 2. 读取 collect 阶段采集的分数数据文件
    input_path = Path(config.data_path)
    if not input_path.exists():
        logger.error(f"找不到数据文件: {input_path}, 请先运行 collect")
        return {}

    with open(input_path, "r", encoding="utf-8") as f:
        collected_data = json.load(f)

    logger.info("=" * 60)
    logger.info("开始搜索最佳阈值组合")
    logger.info("=" * 60)

    # 3. 初始化搜索状态
    best_score = -1       # 当前最高准确率
    best_params = {}      # 当前最优阈值组合
    all_results = []      # 所有组合的评估结果
    count = 0             # 已评估的组合数

    # 4. 三层嵌套循环：穷举所有阈值组合
    for confirm_th in config.confirm_range:
        for options_th in config.options_range:
            # 约束: options 阈值必须小于 confirmed 阈值，否则跳过
            if options_th >= confirm_th:
                continue

            for gap_th in config.gap_range:
                count += 1

                # 4.1 用当前阈值组合评估所有测试用例，得到准确率
                accuracy, details = _evaluate_thresholds(
                    collected_data, confirm_th, options_th, gap_th
                )

                # 4.2 记录该组合的评估结果
                all_results.append({
                    "confirm": confirm_th,
                    "options": options_th,
                    "gap": gap_th,
                    "accuracy": accuracy,
                })

                # 4.3 如果准确率更高，更新最优参数
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {
                        "confirm": confirm_th,
                        "options": options_th,
                        "gap": gap_th
                    }

    # 5. 按准确率降序排序
    all_results.sort(key=lambda x: x["accuracy"], reverse=True)

    # 6. 打印搜索结果
    logger.info(f"\n搜索完成，共评估 {count} 种组合\n")
    _print_top_results(all_results, config.top_n)

    # 7. 打印最佳参数下每条用例的详细对比
    _print_best_details(collected_data, best_params)

    # 8. 保存搜索结果到 JSON 文件（供 verify 阶段读取）
    result_data = {
        "best_params": best_params,
        "best_accuracy": best_score,
        "top_results": all_results[:config.top_n]
    }

    output_path = Path(config.result_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n搜索结果已保存到: {output_path.absolute()}")

    return result_data


def _evaluate_thresholds(
        collected_data: List[Dict], confirm_th: float, options_th: float, gap_th: float
) -> Tuple[float, List[Dict]]:
    """用指定阈值组合评估所有测试用例，返回准确率和逐条详情"""

    correct = 0    # 正确的用例数
    details = []   # 每条用例的详细评估结果

    # 1. 遍历每个测试用例
    for case in collected_data:
        search_results = case.get("search_results", [])
        expected_confirmed = case.get("expected_confirmed", [])
        expected_options = case.get("expected_options", [])

        # 2. 如果该用例没有搜索结果（如 llm_extract 为空），直接视为两个容器都为空
        if not search_results:
            actual_confirmed, actual_options = [], []
        else:
            # 3. 用当前阈值模拟评分对齐逻辑，得到 confirmed 和 options
            actual_confirmed, actual_options = simulate_align(
                search_results, confirm_th, options_th
            )
            # 4. 如果 confirmed 中有多个商品，用 gap 阈值做分数差过滤
            if len(actual_confirmed) > 1:
                actual_confirmed = simulate_filter(actual_confirmed, search_results, gap_th)

        # 5. 对比实际结果和期望结果，判断是否正确
        is_correct = check_result(
            actual_confirmed, expected_confirmed, actual_options, expected_options
        )
        if is_correct:
            correct += 1

        # 6. 记录该用例的详细对比信息
        details.append({
            "query": case["query"],
            "description": case["description"],
            "expected_confirmed": expected_confirmed,
            "actual_confirmed": actual_confirmed,
            "expected_options": expected_options,
            "actual_options": actual_options,
            "correct": is_correct
        })

    # 7. 计算准确率（正确数 / 总数 × 100）
    accuracy = (correct / len(collected_data)) * 100 if collected_data else 0
    return accuracy, details


def _print_top_results(all_results: List[Dict], top_n: int) -> None:
    """打印准确率最高的 Top N 阈值组合"""

    logger.info("=" * 60)
    logger.info(f"Top {top_n} 阈值组合")
    logger.info("=" * 60)

    # 1. 打印表头
    logger.info(f"  {'排名':>4s}  {'confirm':>8s}  {'options':>7s}  {'gap':>5s}  {'准确率':>6s}")
    logger.info(f"  {'─' * 4}  {'─' * 8}  {'─' * 7}  {'─' * 5}  {'─' * 6}")

    # 2. 逐行打印每个组合的参数和准确率
    for rank, r in enumerate(all_results[:top_n], 1):
        logger.info(
            f"  {rank:>4d}  {r['confirm']:>8.2f}  {r['options']:>7.2f}  {r['gap']:>5.2f}  {r['accuracy']:>5.1f}%"
        )


def _print_best_details(collected_data: List[Dict], best_params: Dict) -> None:
    """打印最佳参数下每条测试用例的详细对比结果"""

    logger.info(f"\n{'=' * 60}")
    logger.info("最佳参数下的逐条结果")
    logger.info(f"{'=' * 60}")

    # 1. 用最佳参数重新评估一遍，拿到逐条详情
    _, details = _evaluate_thresholds(
        collected_data, best_params["confirm"], best_params["options"], best_params["gap"]
    )

    # 2. 逐条打印：✓ 表示正确，✕ 表示不匹配
    for d in details:
        status = "✓" if d["correct"] else "✕"
        logger.info(f"\n  {status} {d['description']}")
        logger.info(f"    问题: {d['query']}")
        logger.info(f"    期望 confirmed: {d['expected_confirmed']}")
        logger.info(f"    实际 confirmed: {d['actual_confirmed']}")
        logger.info(f"    期望 options:   {d['expected_options']}")
        logger.info(f"    实际 options:   {d['actual_options']}")

    # 3. 打印最佳参数汇总
    logger.info(f"\n最佳参数:")
    logger.info(f"  confirm 阈值: {best_params['confirm']}")
    logger.info(f"  options 阈值: {best_params['options']}")
    logger.info(f"  gap 阈值:     {best_params['gap']}")


if __name__ == "__main__":
    search()