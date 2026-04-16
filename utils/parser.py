# src/utils/parser.py
import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def robust_parse(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    兼容更多模型输出格式的动作提取器。
    """
    # 1. 预清洗：统一符号，保留代码块内容本身
    text = raw_text.replace('，', ',').replace('【', '[').replace('】', ']').replace('‘', "'").replace('’', "'")
    text = text.replace('```python', '').replace('```json', '').replace('```', '')

    # 2. 定位动作区块
    action_block = text
    if "[Action]" in text:
        action_block = text.split("[Action]")[-1]
    elif "Action:" in text:
        action_block = text.split("Action:")[-1]

    upper = action_block.upper()

    # 1) TYPE
    if "TYPE" in upper:
        kv_match = re.search(r"TYPE\s*[:=]\s*\{[^}]*?(?:text|content)\s*[:=]\s*['\"](.*?)['\"]", action_block,
                             re.IGNORECASE)
        if kv_match:
            return "TYPE", {"content": kv_match.group(1)}  # text 改为 content
        type_match = re.search(r"TYPE.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if type_match:
            return "TYPE", {"content": type_match.group(1)}  # text 改为 content
        type_tail = re.search(r"TYPE\s*[:=]\s*\[?\s*'?\"?(.*?)'?\"?\s*]?$", action_block, re.IGNORECASE)
        if type_tail:
            return "TYPE", {"content": type_tail.group(1).strip("[]'\" ")}  # text 改为 content

    # 2) COMPLETE / ENTER
    if "COMPLETE" in upper:
        return "COMPLETE", {}

    # 3) SCROLL
    if "SCROLL" in upper:
        # 【核心修复 1】：强制带上 SCROLL 前缀，防止抓取到 Analyze 中的无关数字
        scroll_match = re.search(r"SCROLL.*?\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\].*?\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
                                 action_block, flags=re.IGNORECASE)
        if scroll_match:
            return "SCROLL", {
                "start_point": [int(scroll_match.group(1)), int(scroll_match.group(2))],
                "end_point": [int(scroll_match.group(3)), int(scroll_match.group(4))],
            }

        # 兜底：仅在 SCROLL 关键词之后寻找连续数字
        scroll_block = action_block[upper.find("SCROLL"):]
        nums = re.findall(r"-?\d+", scroll_block)
        if len(nums) >= 4:
            return "SCROLL", {
                "start_point": [int(nums[0]), int(nums[1])],
                "end_point": [int(nums[2]), int(nums[3])],
            }

    # 4) OPEN
    if "OPEN" in upper:
        open_match = re.search(r"OPEN.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if open_match:
            return "OPEN", {"app_name": open_match.group(1)}
        app_match = re.search(r"OPEN\s*[:=]\s*\{?.*?(?:app_name|app|content).*?[:=]\s*['\"]?(.*?)['\"]?\s*}?",
                              action_block, re.IGNORECASE)
        if app_match:
            return "OPEN", {"app_name": app_match.group(1).strip()}

    # 5) CLICK / CLICK_ID
    if "CLICK_ID" in upper:
        id_match = re.search(r"CLICK_ID\s*[:=]\s*\[?\s*(\d+)\s*]?", action_block, re.IGNORECASE)
        if id_match:
            return "CLICK_ID", {"id": int(id_match.group(1))}

    if "CLICK" in upper:
        id_match = re.search(r"CLICK\s*[:=]\s*\[?\s*(\d+)\s*]?", action_block, re.IGNORECASE)
        if id_match and "CLICK_ID" not in upper:
            return "CLICK_ID", {"id": int(id_match.group(1))}

        xy_match = re.search(r"(?:x\s*[:=]\s*(-?\d+)).*?(?:y\s*[:=]\s*(-?\d+))", action_block, re.IGNORECASE)
        if xy_match:
            return "CLICK", {"point": [int(xy_match.group(1)), int(xy_match.group(2))]}

        # 【核心修复 2】：严格要求匹配 CLICK 前缀，彻底阻断 Analyze 区块中的坐标干扰
        exact_match = re.search(r"CLICK.*?\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", action_block, flags=re.IGNORECASE)
        if exact_match:
            return "CLICK", {"point": [int(exact_match.group(1)), int(exact_match.group(2))]}

        # 【核心修复 3】：泛数字提取兜底也必须在 CLICK 之后，防止抓错
        click_nums_match = re.search(r"CLICK.*?(-?\d+).*?(-?\d+)", action_block, flags=re.IGNORECASE)
        if click_nums_match:
            return "CLICK", {"point": [int(click_nums_match.group(1)), int(click_nums_match.group(2))]}

    logger.warning(f"[解析失败兜底] 原始输出: {action_block}")
    return "CLICK", {"point": [500, 500]}