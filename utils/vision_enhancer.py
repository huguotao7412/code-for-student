from PIL import Image, ImageDraw


def add_coordinate_grid(image: Image.Image) -> Image.Image:
    """
    在传入的截图上绘制 10x10 的红色半透明参考网格。
    这能极大地帮助大模型在二维空间中建立准确的 1000x1000 归一化位置感知。
    """
    # 转换为 RGBA 以支持透明度叠加
    img_copy = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_copy.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = img_copy.size

    # 绘制 10x10 网格 (每 10% 绘制一条线，对应归一化坐标的 100, 200...900)
    for i in range(1, 10):
        x = int(width * i / 10.0)
        y = int(height * i / 10.0)

        # 垂直线
        draw.line([(x, 0), (x, height)], fill=(255, 0, 0, 80), width=2)
        # 水平线
        draw.line([(0, y), (width, y)], fill=(255, 0, 0, 80), width=2)

    # 合并图像并转回模型支持的 RGB
    result = Image.alpha_composite(img_copy, overlay)
    return result.convert("RGB")