# 文件路径: utils/ui_detector.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


def _pick_label_anchor(x: int, y: int, w: int, h: int, img_w: int, img_h: int, used: set) -> tuple[int, int]:
    """把标签优先放到控件外侧，减少覆盖关键文本。"""
    candidates = [
        (x, max(2, y - 14)),
        (min(img_w - 22, x + w + 3), y),
        (x, min(img_h - 14, y + h + 3)),
        (max(2, x - 20), y),
    ]
    for tx, ty in candidates:
        key = (int(tx / 8), int(ty / 8))
        if key not in used:
            used.add(key)
            return tx, ty
    tx, ty = max(2, x), max(2, y)
    used.add((int(tx / 8), int(ty / 8)))
    return tx, ty


def draw_som_labels(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    给识别到的 UI 元素画框并打上数字编号（低污染版本）。
    返回：(标注后的图片, 编号到元数据的映射表)
    元数据格式：{'center': [x, y], 'bbox': [x0, y0, x1, y1]}
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

    # 编号稳定：先上后下、先左后右、再按面积（大控件优先）
    candidates.sort(key=lambda item: (item[0], item[1], -item[4]))

    element_map = {}
    used_label_slots = set()
    idx = 1
    for y, x, w, h, _ in candidates:
        center_x = int((x + w / 2) * 1000 / img_w)
        center_y = int((y + h / 2) * 1000 / img_h)
        bbox = [
            int(x * 1000 / img_w),
            int(y * 1000 / img_h),
            int((x + w) * 1000 / img_w),
            int((y + h) * 1000 / img_h),
        ]

        # 低污染: 只画细描边，不做大面积半透明填充。
        draw.rectangle([x, y, x + w, y + h], outline=(255, 228, 80, 190), width=1)

        label_text = str(idx)
        tx, ty = _pick_label_anchor(x, y, w, h, img_w, img_h, used_label_slots)
        draw.rectangle([tx - 1, ty - 1, tx + 16, ty + 12], fill=(255, 228, 80, 210))
        draw.text((tx + 2, ty), label_text, fill=(0, 0, 0, 255))

        element_map[idx] = {
            "center": [center_x, center_y],
            "bbox": bbox,
        }
        idx += 1
        if idx > 80:
            break

    return draw_img.convert("RGB"), element_map
