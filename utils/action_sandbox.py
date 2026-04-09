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

        # 尝试吸附：如果点击点在某个识别到的元素 30 像素范围内，自动吸附到中心
        for idx, center in element_map.items():
            dist = ((final_x - center[0]) ** 2 + (final_y - center[1]) ** 2) ** 0.5
            if dist < 35:
                final_x, final_y = center
                break

        params["point"] = [final_x, final_y]

    elif action == "SCROLL":
        if "start_point" in params:
            params["start_point"] = [clip(params["start_point"][0]), clip(params["start_point"][1])]
        if "end_point" in params:
            params["end_point"] = [clip(params["end_point"][0]), clip(params["end_point"][1])]

    return action, params