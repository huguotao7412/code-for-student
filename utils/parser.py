# src/utils/parser.py
import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def robust_parse(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    """
    零容错正则提取器：处理大模型可能输出的不规则坐标和符号
    """
    # 1. 字符级预清洗：统一标点符号，剔除可能的 Markdown 代码块
    text = raw_text.replace('，', ',').replace('【', '[').replace('】', ']').replace('‘', "'").replace('’', "'")
    text = re.sub(r'```[\s\S]*?```', '', text)

    # 2. 定位最终动作块：优先从 [Action] 标记后提取
    action_block = text
    if "[Action]" in text:
        action_block = text.split("[Action]")[-1]
    elif "Action:" in text:
        action_block = text.split("Action:")[-1]

    upper = action_block.upper()

    # 只保留 TestRunner 支持的动作
    if "COMPLETE" in upper:
        return "COMPLETE", {}

    if "SCROLL" in upper:
        nums = re.findall(r"-?\d+", action_block)
        if len(nums) >= 4:
            return "SCROLL", {
                "start_point": [int(nums[0]), int(nums[1])],
                "end_point": [int(nums[2]), int(nums[3])]
            }

    if "TYPE" in upper:
        type_match = re.search(r"TYPE.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if type_match:
            return "TYPE", {"text": type_match.group(1)}
        type_tail = re.search(r"TYPE\s*[:=]\s*(.*)$", action_block, re.IGNORECASE)
        if type_tail:
            return "TYPE", {"text": type_tail.group(1).strip()}

    if "OPEN" in upper:
        open_match = re.search(r"OPEN.*?['\"](.*?)['\"]", action_block, re.IGNORECASE)
        if open_match:
            return "OPEN", {"app_name": open_match.group(1)}
        app_match = re.search(r"OPEN\s*[:=]\s*\{?\s*'?app(?:_name)?'?\s*[:=]\s*['\"]?(.*?)['\"]?\s*}?", action_block, re.IGNORECASE)
        if app_match:
            return "OPEN", {"app_name": app_match.group(1).strip()}

    if "CLICK" in upper:
        # 优先抓取第一个坐标对；允许括号、冒号、空格混排
        click_nums = re.findall(r"-?\d+", action_block)
        if len(click_nums) >= 2:
            return "CLICK", {"point": [int(click_nums[0]), int(click_nums[1])]}

    # 4. 终极兜底：防止解析异常导致评测脚本崩溃得 0 分
    logger.warning(f"[解析失败兜底] 原始输出: {action_block}")
    return "CLICK", {"point": [500, 500]}