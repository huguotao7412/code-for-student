# src/utils/parser.py
import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def robust_parse(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    零容错正则提取器：处理大模型可能输出的不规则坐标和符号
    """
    # 1. 字符级预清洗
    text = raw_text.replace('，', ',').replace('【', '[').replace('】', ']').replace('‘', "'").replace('’', "'")
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 2. 定位最终动作块
    action_block = text
    if "[Action]" in text:
        action_block = text.split("[Action]")[-1]
    elif "Action:" in text:
        action_block = text.split("Action:")[-1]

    upper = action_block.upper()

    # 1. 提取 TYPE
    if "TYPE" in upper:
        type_match = re.search(r"TYPE.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if type_match:
            return "TYPE", {"text": type_match.group(1)}
        type_tail = re.search(r"TYPE\s*[:=]\s*\[?\s*'?\"?(.*?)'?\"?\s*]?$", action_block, re.IGNORECASE)
        if type_tail:
            return "TYPE", {"text": type_tail.group(1).strip("[]'\" ")}

    # 2. 提取 COMPLETE 及 ENTER
    if "COMPLETE" in upper:
        return "COMPLETE", {}
    if "ENTER" in upper:
        return "ENTER", {}

    # 3. 提取 SCROLL
    if "SCROLL" in upper:
        nums = re.findall(r"-?\d+", action_block)
        if len(nums) >= 4:
            return "SCROLL", {
                "start_point": [int(nums[0]), int(nums[1])],
                "end_point": [int(nums[2]), int(nums[3])]
            }

    # 4. 提取 OPEN
    if "OPEN" in upper:
        open_match = re.search(r"OPEN.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if open_match:
            return "OPEN", {"app_name": open_match.group(1)}
        app_match = re.search(r"OPEN\s*[:=]\s*\{?.*?app.*?[:=]\s*['\"]?(.*?)['\"]?\s*}?", action_block, re.IGNORECASE)
        if app_match:
            return "OPEN", {"app_name": app_match.group(1).strip()}

    # 5. 提取 CLICK (增加对 SOM 标签 ID 的支持)
    if "CLICK" in upper:
        # 优先匹配数字标签格式，如 CLICK:[15]
        id_match = re.search(r"CLICK\s*[:=]\s*\[?\s*(\d+)\s*]?", action_block, re.IGNORECASE)
        if id_match:
            return "CLICK_ID", {"id": int(id_match.group(1))}

        # 匹配标准坐标格式 [[x, y]]
        exact_match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*]", action_block)
        if exact_match:
            return "CLICK", {"point": [int(exact_match.group(1)), int(exact_match.group(2))]}

        # 兜底匹配任意两个数字
        click_nums = re.findall(r"-?\d+", action_block)
        if len(click_nums) >= 2:
            return "CLICK", {"point": [int(click_nums[0]), int(click_nums[1])]}

    # 全局兜底
    logger.warning(f"[解析失败兜底] 原始输出: {action_block}")
    return "CLICK", {"point": [500, 500]}