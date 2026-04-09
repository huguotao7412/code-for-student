# 文件路径: utils/visual_memory.py
from PIL import Image, ImageDraw


def draw_previous_action(image: Image.Image, history_actions: list) -> Image.Image:
    """
    在图片上绘制上一步的动作轨迹，形成视觉记忆
    """
    if not history_actions:
        return image

    # 获取上一步动作
    last_action = history_actions[-1]
    action_type = last_action.get('action')
    params = last_action.get('parameters', {})

    img_copy = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = img_copy.size

    def draw_mark(x_norm, y_norm, color=(0, 255, 0, 180)):
        # 归一化坐标转为像素坐标
        x_px = int(x_norm * width / 1000.0)
        y_px = int(y_norm * height / 1000.0)
        r = 12  # 准星半径
        # 画一个半透明的绿色准星
        draw.line([(x_px - r, y_px), (x_px + r, y_px)], fill=color, width=4)
        draw.line([(x_px, y_px - r), (x_px, y_px + r)], fill=color, width=4)
        draw.ellipse([(x_px - r // 2, y_px - r // 2), (x_px + r // 2, y_px + r // 2)], outline=color, width=2)

    if action_type == 'CLICK' and 'point' in params:
        x, y = params['point']
        draw_mark(x, y, color=(0, 255, 0, 180))  # 绿色准星代表上次点击

    elif action_type == 'SCROLL' and 'start_point' in params and 'end_point' in params:
        sx, sy = params['start_point']
        ex, ey = params['end_point']
        sx_px, sy_px = int(sx * width / 1000.0), int(sy * height / 1000.0)
        ex_px, ey_px = int(ex * width / 1000.0), int(ey * height / 1000.0)

        # 绘制滑动轨迹的起点和终点
        draw_mark(sx, sy, color=(0, 150, 255, 180))  # 蓝色起点
        draw_mark(ex, ey, color=(255, 165, 0, 180))  # 橙色终点
        # 绘制滑动箭头连线
        draw.line([(sx_px, sy_px), (ex_px, ey_px)], fill=(255, 165, 0, 150), width=4)

    # 合并图像
    result = Image.alpha_composite(img_copy, overlay)
    return result.convert("RGB")