# 文件路径: utils/action_sandbox.py

def sanitize_and_stick(action: str, params: dict, element_map: dict) -> tuple:
    """
    1. 坐标裁剪 2. 边缘保护 3. 元素吸附(Sticky Click)
    """

    def clip(v):
        return max(10, min(v, 990))

    def extract_meta(meta):
        """兼容旧版 center 结构和新版 bbox 元数据。"""
        if isinstance(meta, dict):
            center = meta.get("center") or meta.get("point")
            bbox = meta.get("bbox")
            return center, bbox
        if isinstance(meta, (list, tuple)):
            if len(meta) == 2:
                return list(meta), None
            if len(meta) == 4:
                x0, y0, x1, y1 = meta
                return [int((x0 + x1) / 2), int((y0 + y1) / 2)], [x0, y0, x1, y1]
        return None, None

    def project_into_bbox(point, bbox, pad=8):
        x0, y0, x1, y1 = bbox
        left = clip(int(x0) + pad)
        top = clip(int(y0) + pad)
        right = clip(int(x1) - pad)
        bottom = clip(int(y1) - pad)
        if left > right:
            mid_x = clip(int((x0 + x1) / 2))
            left = right = mid_x
        if top > bottom:
            mid_y = clip(int((y0 + y1) / 2))
            top = bottom = mid_y
        return [min(max(point[0], left), right), min(max(point[1], top), bottom)]

    def rect_distance(point, bbox):
        x, y = point
        x0, y0, x1, y1 = bbox
        dx = 0 if x0 <= x <= x1 else min(abs(x - x0), abs(x - x1))
        dy = 0 if y0 <= y <= y1 else min(abs(y - y0), abs(y - y1))
        return (dx * dx + dy * dy) ** 0.5

    if action == "CLICK" and "point" in params:
        raw_x, raw_y = params["point"]
        final_x, final_y = clip(raw_x), clip(raw_y)

        best_point = None
        best_score = None
        max_project_distance = 260

        for _, meta in element_map.items():
            center, bbox = extract_meta(meta)
            if not bbox and center:
                bbox = [center[0], center[1], center[0], center[1]]
            if not bbox:
                continue

            dist = rect_distance([final_x, final_y], bbox)
            if dist <= max_project_distance:
                projected = project_into_bbox([final_x, final_y], bbox, pad=8)
                score = dist
                if best_score is None or score < best_score:
                    best_score = score
                    best_point = projected

        if best_point is not None:
            final_x, final_y = best_point
        else:
            # 保守兜底：若没有任何 bbox 足够接近，再尝试中心吸附
            for _, meta in element_map.items():
                center, _ = extract_meta(meta)
                if not center:
                    continue
                dist = ((final_x - center[0]) ** 2 + (final_y - center[1]) ** 2) ** 0.5
                if dist <= 30:
                    final_x, final_y = center
                    break

        params["point"] = [clip(final_x), clip(final_y)]

    elif action == "SCROLL":
        if "start_point" in params:
            params["start_point"] = [clip(params["start_point"][0]), clip(params["start_point"][1])]
        if "end_point" in params:
            params["end_point"] = [clip(params["end_point"][0]), clip(params["end_point"][1])]

    return action, params