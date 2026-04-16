# 文件路径: utils/ui_detector.py
import cv2
import numpy as np
from PIL import Image, ImageDraw


def draw_som_labels(image: Image.Image) -> tuple[Image.Image, dict]:
    """
    升级版 SoM (Set-of-Mark) 标注器
    加入了图像膨胀(Dilation)与层级过滤，防止屏幕被密集的文字碎框淹没。
    """
    # 强制转换为 RGB，防止截图带有 Alpha 透明通道导致 cvtColor 崩溃
    cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # 1. 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 2. 图像膨胀 (核心优化)
    # 将相邻极近的边缘（如一行文字的各个字母）连成一个整体块，避免碎框
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # 3. 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    img_w, img_h = image.size

    # 4. 提取并过滤异常大小的框
    raw_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # 过滤掉太小（噪点）和太大（整个背景块）的区域
        if 25 < w < img_w * 0.85 and 25 < h < img_h * 0.85:
            raw_boxes.append([x, y, x + w, y + h])

    # 5. 简单去重与包含过滤 (去除大框里套着的小框)
    # 按面积从大到小排序
    raw_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    filtered_boxes = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box
        is_inside = False
        for fb in filtered_boxes:
            fx1, fy1, fx2, fy2 = fb
            # 宽容的包含判定 (稍微放大外框的判定范围)
            if x1 >= fx1 - 5 and y1 >= fy1 - 5 and x2 <= fx2 + 5 and y2 <= fy2 + 5:
                is_inside = True
                break
        if not is_inside:
            filtered_boxes.append(box)

    # 6. 让编号排序符合人类阅读习惯：先上后下、再左后右
    filtered_boxes.sort(key=lambda b: (b[1] // 20, b[0]))

    element_map = {}
    idx = 1

    # 定义高对比度颜色：边框用半透明蓝绿，背景黑底白字极其醒目
    box_color = (0, 255, 255, 180)
    tag_bg_color = (0, 0, 0, 220)
    tag_text_color = (255, 255, 255)

    for x1, y1, x2, y2 in filtered_boxes:
        w = x2 - x1
        h = y2 - y1

        # 换算回 0-1000 的归一化坐标系，存入映射表
        center_x = int((x1 + w / 2) * 1000 / img_w)
        center_y = int((y1 + h / 2) * 1000 / img_h)
        element_map[idx] = [center_x, center_y]

        # 绘制识别框
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        label_text = str(idx)
        try:
            # 兼容不同版本 PIL 的文字大小获取方式
            if hasattr(draw, "textbbox"):
                text_bbox = draw.textbbox((0, 0), label_text)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            else:
                text_w, text_h = draw.textsize(label_text)
        except Exception:
            text_w, text_h = 10, 10

        # 在框的左上角或中心偏上方绘制带黑色背景的数字，保证任何底色下都清晰可见
        tag_x = x1 + 2
        tag_y = y1 + 2

        draw.rectangle(
            [tag_x - 1, tag_y - 1, tag_x + text_w + 3, tag_y + text_h + 3],
            fill=tag_bg_color
        )
        draw.text((tag_x + 1, tag_y), label_text, fill=tag_text_color)

        idx += 1

    return draw_img, element_map