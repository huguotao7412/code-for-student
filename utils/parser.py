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
        kv_match = re.search(r"TYPE\s*[:=]\s*\{[^}]*?(?:text|content)\s*[:=]\s*['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if kv_match:
            return "TYPE", {"text": kv_match.group(1)}
        type_match = re.search(r"TYPE.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if type_match:
            return "TYPE", {"text": type_match.group(1)}
        type_tail = re.search(r"TYPE\s*[:=]\s*\[?\s*'?\"?(.*?)'?\"?\s*]?$", action_block, re.IGNORECASE)
        if type_tail:
            return "TYPE", {"text": type_tail.group(1).strip("[]'\" ")}

    # 2) COMPLETE / ENTER
    if "COMPLETE" in upper:
        return "COMPLETE", {}
    if "ENTER" in upper:
        return "ENTER", {}

    # 3) SCROLL
    if "SCROLL" in upper:
        nums = re.findall(r"-?\d+", action_block)
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
        app_match = re.search(r"OPEN\s*[:=]\s*\{?.*?(?:app_name|app|content).*?[:=]\s*['\"]?(.*?)['\"]?\s*}?", action_block, re.IGNORECASE)
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

        exact_match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*]", action_block)
        if exact_match:
            return "CLICK", {"point": [int(exact_match.group(1)), int(exact_match.group(2))]}

        click_nums = re.findall(r"-?\d+", action_block)
        if len(click_nums) >= 2:
            return "CLICK", {"point": [int(click_nums[0]), int(click_nums[1])]}

    logger.warning(f"[解析失败兜底] 原始输出: {action_block}")
    return "CLICK", {"point": [500, 500]}