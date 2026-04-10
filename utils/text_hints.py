# 文件路径: utils/text_hints.py
from dataclasses import dataclass
from typing import List, Any
import logging

from PIL import Image

logger = logging.getLogger(__name__)

# Optional dependency: keep static analyzers happy even when pytesseract is absent.
pytesseract: Any = None
Output: Any = None

try:
    import pytesseract as _pytesseract  # type: ignore
    from pytesseract import Output as _Output  # type: ignore
    pytesseract = _pytesseract
    Output = _Output
except Exception:
    pytesseract = None
    Output = None


@dataclass
class TextHint:
    text: str
    bbox: List[int]
    confidence: float


_GENERIC_UI_WORDS = [
    "搜索", "搜索框", "查找", "输入", "确认", "确定", "发送", "提交",
    "返回", "关闭", "取消", "跳过", "同意", "完成", "完成", "登录", "注册",
]


def extract_text_hints(image: Image.Image, max_hints: int = 24) -> List[TextHint]:
    """提取轻量文本提示；OCR 不可用时返回空列表。"""
    if pytesseract is None or Output is None:
        return []

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--psm 6")
    except Exception as e:
        logger.debug(f"OCR unavailable: {e}")
        return []

    texts = data.get("text", [])
    confs = data.get("conf", [])
    lefts = data.get("left", [])
    tops = data.get("top", [])
    widths = data.get("width", [])
    heights = data.get("height", [])

    hints: List[TextHint] = []
    for i, raw_text in enumerate(texts):
        text = (raw_text or "").strip()
        if not text:
            continue
        try:
            conf = float(confs[i])
        except Exception:
            conf = -1.0
        if conf < 35:
            continue

        x0 = int(lefts[i])
        y0 = int(tops[i])
        x1 = int(lefts[i] + widths[i])
        y1 = int(tops[i] + heights[i])
        hints.append(TextHint(text=text, bbox=[x0, y0, x1, y1], confidence=max(0.0, min(conf, 100.0)) / 100.0))
        if len(hints) >= max_hints:
            break

    return hints


def summarize_text_hints(hints: List[TextHint], limit: int = 6) -> str:
    """压缩成适合 prompt 的短摘要。"""
    if not hints:
        return ""

    ranked = []
    for hint in hints:
        text = hint.text.strip()
        if not text:
            continue
        keyword_boost = 0.0
        if any(k in text for k in _GENERIC_UI_WORDS):
            keyword_boost += 1.0
        if len(text) <= 6:
            keyword_boost += 0.2
        ranked.append((hint.confidence + keyword_boost, hint))

    ranked.sort(key=lambda item: item[0], reverse=True)
    parts = []
    for _, hint in ranked[:limit]:
        x0, y0, x1, y1 = hint.bbox
        parts.append(f"{hint.text}@({x0},{y0},{x1},{y1})")
    return "；".join(parts)


def keyword_hit_score(text: str, keywords: List[str]) -> float:
    """辅助词命中分数，便于 Agent 做候选排序。"""
    if not text:
        return 0.0
    score = 0.0
    for kw in keywords:
        if kw and kw in text:
            score += max(1.0, len(kw) * 0.8)
    return score
