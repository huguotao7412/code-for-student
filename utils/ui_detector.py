# 文件路径: utils/ui_detector.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


def _pick_label_anchor(x: int, y: int, w: int, h: int, img_w: int, img_h: int, used: set) -> tuple[int, int]:
    """把标签优先放到控件外部安全区域，彻底避免污染控件内部。"""
    # 扩大搜索候选点，优先放左上角外侧
    candidates = [
        (max(2, x - 20), max(2, y - 20)),
        (min(img_w - 22, x + w + 5), max(2, y - 20)),
        (max(2, x - 20), min(img_h - 14, y + h + 5)),
        (min(img_w - 22, x + w + 5), min(img_h - 14, y + h + 5)),
    ]
    for tx, ty in candidates:
        key = (int(tx / 15), int(ty / 15))  # 放宽防碰撞网格
        if key not in used:
            used.add(key)
            return tx, ty
    tx, ty = max(2, x - 10), max(2, y - 10)
    used.add((int(tx / 15), int(ty / 15)))
    return tx, ty


def draw_som_labels(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    极低污染版 SoM 标记算法：
    放弃全包围边框，改用轻量化护角，并通过指示线将数字标号拉出控件外。
    """
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(draw_img)
    img_w, img_h = image.size

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 22 < w < img_w * 0.92 and 22 < h < img_h * 0.92 and area > 900:
            candidates.append((y, x, w, h, area))

    candidates.sort(key=lambda item: (item[0], item[1], -item[4]))

    element_map = {}
    used_label_slots = set()
    idx = 1

    # 极轻量绘制色系
    corner_color = (135, 206, 250, 200)  # 浅蓝色护角
    label_bg = (255, 228, 80, 220)  # 明黄色气泡

    for y, x, w, h, _ in candidates:
        center_x = int((x + w / 2) * 1000 / img_w)
        center_y = int((y + h / 2) * 1000 / img_h)
        bbox = [
            int(x * 1000 / img_w), int(y * 1000 / img_h),
            int((x + w) * 1000 / img_w), int((y + h) * 1000 / img_h),
        ]

        # 1. 不再画全包裹矩形框，只画 L 型轻量级护角 (长度为边长的20%，最长不超过15px)
        c_len = min(15, max(4, int(min(w, h) * 0.2)))
        # 左上
        draw.line([(x, y + c_len), (x, y), (x + c_len, y)], fill=corner_color, width=2)
        # 右下
        draw.line([(x + w - c_len, y + h), (x + w, y + h), (x + w, y + h - c_len)], fill=corner_color, width=2)

        # 2. 计算标签抛出点，并绘制半透明牵引线
        tx, ty = _pick_label_anchor(x, y, w, h, img_w, img_h, used_label_slots)
        draw.line([(x, y), (tx + 8, ty + 6)], fill=(255, 255, 255, 120), width=1)

        # 3. 绘制外部标签泡泡
        label_text = str(idx)
        # 增加一点阴影让标签从画面中浮出
        draw.rectangle([tx, ty, tx + 18, ty + 14], fill=(0, 0, 0, 80))
        draw.rectangle([tx - 1, ty - 1, tx + 17, ty + 13], fill=label_bg)
        draw.text((tx + 2, ty), label_text, fill=(0, 0, 0, 255))

        element_map[idx] = {
            "center": [center_x, center_y],
            "bbox": bbox,
        }
        idx += 1
        if idx > 80:
            break

    return draw_img.convert("RGB"), element_map