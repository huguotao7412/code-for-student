# 文件路径: utils/ui_detector.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_som_labels(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    给识别到的 UI 元素画框并打上数字编号
    返回：(标注后的图片, 编号到归一化坐标的映射表)
    """
    # PIL 转 CV2
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 边缘检测与轮廓寻找
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    element_map = {}
    idx = 1

    img_w, img_h = image.size
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤过小或过大的噪点
        if 20 < w < img_w * 0.8 and 20 < h < img_h * 0.8:
            # 计算归一化中心点 (0-1000)
            center_x = int((x + w / 2) * 1000 / img_w)
            center_y = int((y + h / 2) * 1000 / img_h)

            # 绘制黄色半透明边框
            draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0, 150), width=2)
            # 绘制编号标签背景
            label_w, label_h = 25, 20
            draw.rectangle([x, y, x + label_w, y + label_h], fill=(255, 255, 0, 200))
            # 绘制黑字编号
            draw.text((x + 5, y + 2), str(idx), fill=(0, 0, 0))

            element_map[idx] = [center_x, center_y]
            idx += 1
            if idx > 80: break  # 防止标注过多干扰模型

    return draw_img, element_map