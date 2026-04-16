import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def add_coordinate_grid(image: Image.Image, grid_size=10, opacity=128) -> Image.Image:
    """
    升级版：边缘标尺法 (Edge Rulers)
    放弃全屏网格，改为在图像四周边缘绘制标尺，并强制将数字刻度映射为 0-1000 归一化坐标。
    绝对不污染原图内部的 UI 像素，极大提升对细小图标的点击精确度。
    """
    try:
        img = image.copy().convert("RGBA")
        width, height = img.size

        # 创建一个透明图层用于绘制标尺
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        ruler_width = max(30, int(min(width, height) * 0.04))
        bg_color = (0, 0, 0, 180)
        fg_color = (255, 255, 255, 255)

        draw.rectangle([0, 0, width, ruler_width], fill=bg_color)  # Top
        draw.rectangle([0, height - ruler_width, width, height], fill=bg_color)  # Bottom
        draw.rectangle([0, 0, ruler_width, height], fill=bg_color)  # Left
        draw.rectangle([width - ruler_width, 0, width, height], fill=bg_color)  # Right

        font_size = max(12, int(ruler_width * 0.5))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        for i in range(0, 1001, 50):
            is_major = (i % 100 == 0)
            tick_len = ruler_width if is_major else ruler_width // 2

            px = int(width * i / 1000)
            py = int(height * i / 1000)

            draw.line([(px, 0), (px, tick_len)], fill=fg_color, width=2)
            if is_major and 0 < i < 1000:
                draw.text((px + 4, 2), str(i), fill=fg_color, font=font)

            draw.line([(px, height), (px, height - tick_len)], fill=fg_color, width=2)
            if is_major and 0 < i < 1000:
                draw.text((px + 4, height - ruler_width + 2), str(i), fill=fg_color, font=font)

            draw.line([(0, py), (tick_len, py)], fill=fg_color, width=2)
            if is_major and 0 < py < height:
                draw.text((2, py + 2), str(i), fill=fg_color, font=font)

            draw.line([(width, py), (width - tick_len, py)], fill=fg_color, width=2)
            if is_major and 0 < py < height:
                draw.text((width - ruler_width + 2, py + 2), str(i), fill=fg_color, font=font)

        return Image.alpha_composite(img, overlay).convert("RGB")

    except Exception as e:
        logger.error(f"边缘标尺渲染失败，降级原图: {e}")
        return image.copy()