# 文件路径: utils/action_sandbox.py

def sanitize_and_stick(action: str, params: dict, element_map: dict) -> tuple:
    """
    1. 坐标裁剪 2. 边缘保护 3. 元素吸附(Sticky Click)
    """

    def clip(v):
        return max(10, min(v, 990))

    if action == "CLICK" and "point" in params:
        raw_x, raw_y = params["point"]
        final_x, final_y = clip(raw_x), clip(raw_y)

        # 【修改点】：将吸附阈值从 35 降低至 20 像素，提升在密集按键区（如键盘）的精度
        for idx, center in element_map.items():
            dist = ((final_x - center[0]) ** 2 + (final_y - center[1]) ** 2) ** 0.5
            if dist <= 20:
                final_x, final_y = center
                break

        params["point"] = [final_x, final_y]

    elif action == "SCROLL":
        if "start_point" in params:
            params["start_point"] = [clip(params["start_point"][0]), clip(params["start_point"][1])]
        if "end_point" in params:
            params["end_point"] = [clip(params["end_point"][0]), clip(params["end_point"][1])]

    return action, params