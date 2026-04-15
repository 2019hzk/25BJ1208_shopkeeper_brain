"""阈值测试 —— 共享配置与工具方法"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """阈值测试配置"""
    # 文件路径
    data_path: str = "threshold_test_data.json"
    result_path: str = "threshold_search_result.json"
    collection_name: str = "kb_item_names_v1"

    # 阈值搜索范围
    confirm_range: List[float] = field(default_factory=lambda: [0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    options_range: List[float] = field(default_factory=lambda: [0.45, 0.50, 0.55, 0.60, 0.65])
    gap_range: List[float] = field(default_factory=lambda: [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25])

    # 显示配置
    top_n: int = 10

    @classmethod
    def from_file(cls, path: str):
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ─────────────────────────────────────────────────────────
# 模拟评分逻辑（search 和 collect 共用）
# ─────────────────────────────────────────────────────────

def simulate_align(
        search_results: List[Dict], confirm_th: float, options_th: float
) -> Tuple[List[str], List[str]]:
    """
    模拟 _ItemNameAligner._align 逻辑（使用可配置阈值）
    """
    confirmed = []
    options = []

    for sr in search_results:
        extracted_name = sr.get("extracted_name")
        matches = sorted(sr.get("matches", []), key=lambda x: x["score"], reverse=True)

        high = [m for m in matches if m.get("score", 0) >= confirm_th]

        if high:
            exact = next((h for h in high if str(h["item_name"]) == extracted_name), None)

            if exact:
                picked = exact["item_name"]
                if picked not in confirmed:
                    confirmed.append(picked)
            elif len(high) == 1:
                picked = high[0]["item_name"]
                if picked not in confirmed:
                    confirmed.append(picked)
            else:
                for h in high[:3]:
                    picked = h.get("item_name")
                    if picked not in options and picked not in confirmed:
                        options.append(picked)
        else:
            mid = [
                m for m in matches
                if m["score"] >= options_th
                   and m.get("item_name") not in options
                   and m.get("item_name") not in confirmed
            ]
            for m in mid[:3]:
                options.append(m.get("item_name"))

    return confirmed, options[:3]


def simulate_filter(
        confirmed: List[str], search_results: List[Dict], gap_th: float
) -> List[str]:
    """
    模拟 _ItemNameAligner._item_name_score_filter 逻辑（使用可配置阈值）
    """
    score_map = {}
    for sr in search_results:
        for m in sr.get("matches", []):
            name = m.get("item_name")
            score = m.get("score", 0)
            if name in confirmed:
                score_map[name] = max(score_map.get(name, 0), score)

    if not score_map:
        return confirmed

    max_score = max(score_map.values())
    return [name for name, score in score_map.items() if max_score - score <= gap_th]


def check_result(
        actual_confirmed: List[str], expected_confirmed: List[str],
        actual_options: List[str], expected_options: List[str]
) -> bool:
    """
    判断结果是否正确：
    - confirmed: 实际结果必须包含所有期望的商品名（允许多，不允许少）
    - options: 只在期望非空时检查
    """
    for expected in expected_confirmed:
        if expected not in actual_confirmed:
            return False

    if not expected_confirmed and actual_confirmed:
        return False

    if expected_options:
        for expected in expected_options:
            if expected not in actual_options:
                return False

    return True