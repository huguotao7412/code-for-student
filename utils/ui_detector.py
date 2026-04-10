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
        label_w, label_h = 25, 20
        draw.rectangle([x, y, x + label_w, y + label_h], fill=(255, 255, 0, 200))
        draw.text((x + 5, y + 2), str(idx), fill=(0, 0, 0))

        element_map[idx] = {
            "center": [center_x, center_y],
            "bbox": bbox,
        }
        idx += 1
        if idx > 80:
            break

    return draw_img, element_map