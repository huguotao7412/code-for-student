# 文件路径: utils/ui_detector.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_som_labels(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    给识别到的 UI 元素画框并打上数字编号
    返回：(标注后的图片, 编号到元数据的映射表)
    元数据格式：{'center': [x, y], 'bbox': [x0, y0, x1, y1]}
    """
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    img_w, img_h = image.size

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < w < img_w * 0.8 and 20 < h < img_h * 0.8:
            candidates.append((y, x, w, h))

    # 让编号尽量稳定：先上后下、再左后右
    candidates.sort(key=lambda item: (item[0], item[1], -item[2] * item[3]))

    element_map = {}
    idx = 1
    for y, x, w, h in candidates:
        center_x = int((x + w / 2) * 1000 / img_w)
        center_y = int((y + h / 2) * 1000 / img_h)
        bbox = [
            int(x * 1000 / img_w),
            int(y * 1000 / img_h),
            int((x + w) * 1000 / img_w),
            int((y + h) * 1000 / img_h),
        ]

        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0, 150), width=2)

        # 把编号画在按钮中心，避免标签偏到角落
        label_text = str(idx)
        try:
            text_bbox = draw.textbbox((0, 0), label_text)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            text_w, text_h = 12, 12
        text_x = x + max(0, int(w / 2 - text_w / 2))
        text_y = y + max(0, int(h / 2 - text_h / 2))
        bg_pad_x = max(4, int(text_w / 2) + 4)
        bg_pad_y = max(4, int(text_h / 2) + 4)
        draw.ellipse([x + w / 2 - bg_pad_x, y + h / 2 - bg_pad_y, x + w / 2 + bg_pad_x, y + h / 2 + bg_pad_y], fill=(255, 255, 0, 220))
        draw.text((text_x, text_y), label_text, fill=(0, 0, 0))

        element_map[idx] = {
            "center": [center_x, center_y],
            "bbox": bbox,
        }
        idx += 1
        if idx > 80:
            break

    return draw_img, element_map