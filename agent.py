# 文件路径: agent.py
from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Dict, Any, Tuple
from agent_base import BaseAgent, AgentOutput, AgentInput

# 导入视觉与后处理工具
from utils.vision_enhancer import add_coordinate_grid
from utils.visual_memory import draw_previous_action
from utils.ui_detector import draw_som_labels
from utils.action_sandbox import sanitize_and_stick
from utils.text_hints import extract_text_hints, summarize_text_hints, keyword_hit_score

# 导入管理工具
from utils.state import GUIState
from utils.parser import robust_parse

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """通用 GUI Agent：模型优先，支持 SOM 标签直点与动态规则。"""

    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}

    def _initialize(self):
        self.state = GUIState(max_steps=45)

    def reset(self):
        self.state = GUIState(max_steps=45)

    def _dynamic_clean_type_text(self, text: str) -> str:
        """保守清洗文本：只去掉多余包裹符号与首尾空白，避免过度裁剪。"""
        if not text:
            return text
        cleaned = text.strip().strip("'\" ")
        cleaned = cleaned.replace("《", "").replace("》", "").strip()
        return cleaned or text

    def _apply_edge_repulsion(self, point: list, max_val: int = 1000, margin: int = 40) -> list:
        """边缘斥力：避免点击过贴边导致误触。"""
        if not point or len(point) != 2:
            return [500, 500]
        x, y = int(point[0]), int(point[1])
        if x > max_val - margin:
            x = max_val - int(margin * 1.5)
        elif x < margin:
            x = int(margin * 1.5)
        if y > max_val - margin:
            y = max_val - int(margin * 1.5)
        elif y < margin:
            y = int(margin * 1.5)
        return [max(10, min(990, x)), max(10, min(990, y))]

    @staticmethod
    def _clip_norm_point(point: list) -> list:
        if not point or len(point) != 2:
            return [500, 500]
        x, y = int(point[0]), int(point[1])
        return [max(10, min(990, x)), max(10, min(990, y))]

    def _normalize_click_point(self, point: list, normalized_map: Dict[int, Any], prefer_top: bool) -> list:
        """将模型点击点稳健归一化到 0-1000，避免极端偏移。"""
        if not point or len(point) != 2:
            return self._pick_click_target(normalized_map, prefer_top=prefer_top) or [500, 500]

        x, y = int(point[0]), int(point[1])
        img_w, img_h = getattr(self, "_current_image_size", (0, 0))

        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return self._clip_norm_point([x, y])

        if img_w > 0 and img_h > 0 and 0 <= x <= img_w and 0 <= y <= img_h:
            return self._clip_norm_point([int(x * 1000 / img_w), int(y * 1000 / img_h)])

        if img_w > 0 and img_h > 0 and x >= 0 and y >= 0 and x <= img_w * 2 and y <= img_h * 2:
            nx = int(min(x, img_w) * 1000 / img_w)
            ny = int(min(y, img_h) * 1000 / img_h)
            return self._clip_norm_point([nx, ny])

        return self._pick_click_target(normalized_map, prefer_top=prefer_top) or [500, 500]

    def _infer_app_name(self, instruction: str) -> str:
        """从指令中尽量泛化地提取要打开的应用名。"""
        text = instruction.strip()
        for prefix in ["打开", "进入", "启动", "去", "到"]:
            if prefix in text:
                tail = text.split(prefix, 1)[1].strip()
                tail = tail.split("，", 1)[0].split(",", 1)[0].split("。", 1)[0].split("；", 1)[0]
                stop_words = ["搜索", "查找", "查看", "播放", "打车", "评论", "发布", "收藏", "输入", "选择", "购买", "下载", "预约", "登录", "注册", "关注", "浏览", "点开", "下单", "确认", "完成"]
                cut = len(tail)
                for w in stop_words:
                    idx = tail.find(w)
                    if idx > 0:
                        cut = min(cut, idx)
                app_name = tail[:cut].strip(" 的里中上")
                if app_name:
                    return app_name
        return ""

    def _get_situational_rules(self, instruction: str, last_action: str) -> str:
        rules = []
        if last_action in ["TYPE", "ENTER"]:
            rules.append("- ⚠️ 刚完成输入时，不要继续输入；优先查找页面上可见的确认/搜索/发送按钮。")
        if any(k in instruction for k in ["搜索", "查找", "输入", "填写", "查询", "评论", "发布", "发送"]):
            rules.append("- 🎯 如果目标是文本输入或检索，先点输入框，再输入目标文本，再点可见确认按钮。")
        if any(k in instruction for k in ["滚动", "翻页", "浏览"]):
            rules.append("- 🌀 如果当前页面信息不足，优先滚动寻找目标，而不是重复点击同一处。")
        return "\n[通用恢复规则]\n" + "\n".join(rules) if rules else ""

    def _build_cot_prompt(self, instruction: str, history_actions: list, text_hint_summary: str = "") -> str:
        history_summary = self.state.get_summary()
        last_action = history_actions[-1].get("action", "").split(":")[0] if history_actions else ""

        dynamic_reminder = ""
        if last_action == "TYPE":
            dynamic_reminder = "\n【强拦截】：上步已 TYPE，此步先确认页面变化，再决定是否点击确认按钮，不要重复 TYPE。"

        situational_rules = self._get_situational_rules(instruction, last_action)
        plan_context = (
            "任务第一步：识别当前应用和可见入口。"
            if self.state.step_count == 0
            else f"【全局计划】:\n{self.state.global_plan}\n继续执行。"
        )
        hint_line = text_hint_summary if text_hint_summary else "无明显文本提示"

        return f"""你是一个高级 Android GUI 智能代理。
任务目标：【{instruction}】

[当前上下文]
1. 历史动作: {history_summary}
2. 当前进度: {self.state.step_count}/{self.state.max_steps}
3. 【视觉提示】：截图已叠加红色数字标签。
4. 【文本提示】：{hint_line}
5. {plan_context}{dynamic_reminder}

[交互铁律]
- 🎯【优先点标签】：如果能看清红色编号标签，优先输出 CLICK:[编号]；这比直接猜坐标更稳。
- 🛑【终点判定】：若画面已显示目标结果，立即输出 COMPLETE:[]！
- ⌨️【输入原则】：先点击输入框，再 TYPE；输入后优先点击页面上的搜索/确认/发送按钮。{situational_rules}
- 🚫 不要输出 ENTER。

请按结构输出：
[Observation] 任务目的是什么？当前是否已达成？
[Thought] 明确下一步意图。
[Action] 选其一：CLICK:[标签编号], CLICK:[[x, y]], TYPE:['文本'], SCROLL:[[x1, y1], [x2, y2]], OPEN:['应用名'], COMPLETE:[]
[Expected Effect] 预期变化。
"""

    def _parse_with_effect(self, raw_text: str) -> Tuple[str, Dict[str, Any], str]:
        action_block = raw_text
        if "[Action]" in raw_text:
            action_block = raw_text.split("[Action]")[-1].split("[Expected Effect]")[0]
        action, params = robust_parse(action_block)
        expected_effect = raw_text.split("[Expected Effect]")[-1].strip() if "[Expected Effect]" in raw_text else "画面发生变化"
        return action, params, expected_effect

    @staticmethod
    def _extract_center(meta: Any) -> list:
        if isinstance(meta, dict):
            center = meta.get("center") or meta.get("point")
            return list(center) if center else []
        if isinstance(meta, (list, tuple)) and len(meta) == 2:
            return list(meta)
        if isinstance(meta, (list, tuple)) and len(meta) == 4:
            x0, y0, x1, y1 = meta
            return [int((x0 + x1) / 2), int((y0 + y1) / 2)]
        return []

    @staticmethod
    def _extract_bbox(meta: Any) -> list:
        if isinstance(meta, dict):
            bbox = meta.get("bbox")
            return list(bbox) if bbox else []
        if isinstance(meta, (list, tuple)) and len(meta) == 4:
            return list(meta)
        return []

    @staticmethod
    def _bbox_distance(point: list, bbox: list) -> float:
        x, y = point
        x0, y0, x1, y1 = bbox
        dx = 0 if x0 <= x <= x1 else min(abs(x - x0), abs(x - x1))
        dy = 0 if y0 <= y <= y1 else min(abs(y - y0), abs(y - y1))
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _project_point_to_bbox(point: list, bbox: list, pad: int = 8) -> list:
        x0, y0, x1, y1 = bbox
        left = min(max(x0 + pad, 10), 990)
        top = min(max(y0 + pad, 10), 990)
        right = min(max(x1 - pad, 10), 990)
        bottom = min(max(y1 - pad, 10), 990)
        if left > right:
            mid_x = min(max(int((x0 + x1) / 2), 10), 990)
            left = right = mid_x
        if top > bottom:
            mid_y = min(max(int((y0 + y1) / 2), 10), 990)
            top = bottom = mid_y
        return [min(max(point[0], left), right), min(max(point[1], top), bottom)]

    @classmethod
    def _best_bbox_target(cls, point: list, normalized_map: Dict[int, Any], prefer_top: bool = False) -> list:
        """根据点击点选择最合适的 bbox，并投影到该框内部。"""
        candidates = []
        for meta in normalized_map.values():
            bbox = cls._extract_bbox(meta)
            center = cls._extract_center(meta)
            if bbox:
                candidates.append((bbox, center))
        if not candidates:
            return []

        ranked = []
        for bbox, center in candidates:
            dist = cls._bbox_distance(point, bbox)
            if center:
                dist += abs(center[0] - point[0]) * 0.03
                if prefer_top:
                    dist += center[1] * 0.02
            ranked.append((dist, bbox))
        ranked.sort(key=lambda item: item[0])

        best_dist, best_bbox = ranked[0]
        if best_dist > 240:
            return []
        return cls._project_point_to_bbox(point, best_bbox, pad=8)

    @classmethod
    def _pick_click_target(cls, normalized_map: Dict[int, Any], prefer_top: bool = False) -> list:
        """从候选元素中挑选点击目标；优先选择 bbox 内部点。"""
        if not normalized_map:
            return []
        if prefer_top:
            best = None
            best_score = None
            for meta in normalized_map.values():
                center = cls._extract_center(meta)
                bbox = cls._extract_bbox(meta)
                if not center:
                    continue
                score = center[1] + abs(center[0] - 500) * 0.25
                if bbox:
                    score += bbox[1] * 0.15
                if best_score is None or score < best_score:
                    best_score = score
                    best = (center, bbox)
            if best is None:
                return []
            center, bbox = best
            return cls._project_point_to_bbox(center, bbox, pad=8) if bbox else list(center)

        best = None
        best_score = None
        for meta in normalized_map.values():
            center = cls._extract_center(meta)
            bbox = cls._extract_bbox(meta)
            if not center:
                continue
            score = abs(center[0] - 500) + center[1] * 0.15
            if bbox:
                score += bbox[0] * 0.02
            if best_score is None or score < best_score:
                best_score = score
                best = (center, bbox)
        if not best:
            return []
        center, bbox = best
        return cls._project_point_to_bbox(center, bbox, pad=8) if bbox else list(center)

    def _top_candidate(self, normalized_map: Dict[int, Any]) -> list:
        """挑选更像输入框/搜索栏的顶部候选点。"""
        if not normalized_map:
            return []
        top = []
        for meta in normalized_map.values():
            center = self._extract_center(meta)
            bbox = self._extract_bbox(meta)
            if center and 20 <= center[1] <= 220:
                top.append((center, bbox))
        if not top:
            return []
        center, bbox = min(top, key=lambda item: (item[0][1], abs(item[0][0] - 500)))
        return self._project_point_to_bbox(center, bbox, pad=8) if bbox else list(center)

    def _looks_like_input_flow(self, instruction: str, history_actions: list) -> bool:
        keywords = ["搜索", "查找", "输入", "填写", "查询", "打车", "评论", "播放", "收藏", "关注", "发布", "发送"]
        if not any(k in instruction for k in keywords):
            return False
        has_typed = any(act.get("action", "").startswith("TYPE") for act in history_actions)
        return not has_typed

    def _normalize_point(self, point: list, width: int, height: int) -> list:
        if not point or len(point) != 2 or width <= 0 or height <= 0:
            return []
        return [int(point[0] * 1000 / width), int(point[1] * 1000 / height)]

    def _normalize_bbox(self, bbox: list, width: int, height: int) -> list:
        if not bbox or len(bbox) != 4 or width <= 0 or height <= 0:
            return []
        x0, y0, x1, y1 = bbox
        return [
            int(x0 * 1000 / width),
            int(y0 * 1000 / height),
            int(x1 * 1000 / width),
            int(y1 * 1000 / height),
        ]

    def _build_candidates(
        self,
        instruction: str,
        history_actions: list,
        normalized_map: Dict[int, Any],
        text_hints: list,
        model_action: str,
        model_params: Dict[str, Any],
        model_effect: str,
        raw_output: str,
    ) -> list:
        """候选生成阶段：把模型、文本提示和结构化规则都转成候选动作。"""
        candidates = []
        last_action = history_actions[-1].get("action", "").split(":")[0] if history_actions else ""
        input_flow = self._looks_like_input_flow(instruction, history_actions)

        def add_candidate(action: str, params: Dict[str, Any], source: str, score: float, reason: str, effect: str = ""):
            params_copy = {}
            for k, v in params.items():
                params_copy[k] = list(v) if isinstance(v, list) else v
            candidates.append({
                "action": action,
                "params": params_copy,
                "source": source,
                "score": score,
                "reason": reason,
                "expected_effect": effect,
                "raw_output": raw_output,
            })

        # 1) 模型候选
        if model_action == "CLICK_ID":
            tid = model_params.get("id")
            if tid in normalized_map:
                add_candidate("CLICK", {"point": self._extract_center(normalized_map[tid])}, "model", 62, "model click id")
            else:
                pt = self._pick_click_target(normalized_map, prefer_top=input_flow)
                add_candidate("CLICK", {"point": pt or [500, 500]}, "model", 44, "model click id fallback")
        elif model_action == "CLICK" and model_params.get("point"):
            pt = model_params.get("point")
            add_candidate("CLICK", {"point": pt}, "model", 65, "model click point", model_effect)
            corrected = self._best_bbox_target(pt, normalized_map, prefer_top=input_flow)
            if corrected and corrected != pt:
                add_candidate("CLICK", {"point": corrected}, "model", 70, "bbox corrected model click", model_effect)
        elif model_action == "TYPE":
            add_candidate("TYPE", {"text": self._dynamic_clean_type_text(model_params.get("text", ""))}, "model", 72, "model type", model_effect)
        elif model_action in {"OPEN", "SCROLL", "COMPLETE"}:
            add_candidate(model_action, model_params, "model", 72, f"model {model_action.lower()}", model_effect)
        elif model_action == "ENTER":
            top_point = self._top_candidate(normalized_map)
            add_candidate("CLICK", {"point": top_point or self._pick_click_target(normalized_map, prefer_top=True) or [500, 500]}, "model", 48, "enter converted to click", model_effect)
        else:
            add_candidate("CLICK", {"point": [500, 500]}, "model", 12, "model parse fallback", model_effect)

        # 2) OCR / 文本提示候选
        search_words = ["搜索", "搜索框", "查找", "输入", "输入框"]
        confirm_words = ["确认", "确定", "发送", "提交", "完成"]
        back_words = ["返回", "关闭", "取消", "跳过", "同意"]
        ocr_triggers = search_words + confirm_words + back_words

        for hint in text_hints:
            hint_text = (hint.text or "").strip()
            if not hint_text:
                continue
            norm_bbox = self._normalize_bbox(hint.bbox, *self._current_image_size)
            if not norm_bbox:
                continue
            center = [int((norm_bbox[0] + norm_bbox[2]) / 2), int((norm_bbox[1] + norm_bbox[3]) / 2)]
            hit_score = keyword_hit_score(hint_text, ocr_triggers)
            if hit_score <= 0 and hint.confidence < 0.45:
                continue

            score = 18 + hit_score * 5 + hint.confidence * 10
            reason = f"ocr:{hint_text}"
            if any(w in hint_text for w in back_words):
                score += 20
                if any(w in instruction for w in back_words):
                    score += 25
            if any(w in hint_text for w in search_words):
                score += 18
                if any(w in instruction for w in search_words):
                    score += 25
                if input_flow:
                    score += 18
            if any(w in hint_text for w in confirm_words):
                score += 14
                if last_action == "TYPE":
                    score += 20
            if center[1] <= 220 and any(w in hint_text for w in ["搜索", "返回", "确认", "发送", "输入"]):
                score += 8

            add_candidate("CLICK", {"point": center}, "ocr", score, reason, f"text hint:{hint_text}")
            if any(w in hint_text for w in ["返回", "关闭", "取消"]):
                add_candidate("CLICK", {"point": [min(max(center[0], 30), 180), min(max(center[1], 30), 180)]}, "ocr", score - 4, reason + " edge", "text hint edge")

        # 3) 结构化候选
        if self.state.step_count == 0:
            app_name = self._infer_app_name(instruction)
            if app_name:
                add_candidate("OPEN", {"app_name": app_name}, "structural", 52, f"infer app {app_name}")

        if input_flow:
            top_point = self._top_candidate(normalized_map)
            if top_point:
                add_candidate("CLICK", {"point": top_point}, "structural", 46, "input-flow top candidate")

        if any(k in instruction for k in ["滚动", "翻页", "浏览", "下滑", "上滑"]):
            add_candidate("SCROLL", {"start_point": [500, 760], "end_point": [500, 260]}, "structural", 34, "browse scroll")

        if any(k in instruction for k in ["返回", "关闭", "取消", "跳过"]):
            add_candidate("CLICK", {"point": [60, 80]}, "structural", 40, "escape/back button")

        add_candidate("CLICK", {"point": self._pick_click_target(normalized_map, prefer_top=input_flow) or [500, 500]}, "fallback", 5, "safe fallback")
        return candidates

    def _rank_candidates(self, candidates: list, instruction: str, history_actions: list) -> dict:
        """候选排序阶段：结合指令意图、历史状态和候选来源打分。"""
        last_action = history_actions[-1].get("action", "").split(":")[0] if history_actions else ""
        input_flow = self._looks_like_input_flow(instruction, history_actions)
        ranked = []

        for idx, cand in enumerate(candidates):
            score = float(cand.get("score", 0.0))
            action = cand.get("action", "")
            params = cand.get("params", {})
            reason = cand.get("reason", "")
            source = cand.get("source", "")
            pt = params.get("point")
            text = str(params.get("text", ""))
            app_name = str(params.get("app_name", ""))

            # 更偏模型优先，避免过多启发式改写模型意图
            if source == "model":
                score += 18
            elif source == "ocr":
                score += 6
            elif source == "structural":
                score += 3

            if action == "CLICK":
                if pt and len(pt) == 2:
                    x, y = pt
                    if any(k in instruction for k in ["返回", "关闭", "取消", "跳过", "同意"]):
                        if x <= 220 and y <= 220:
                            score += 24
                    if input_flow and y <= 260:
                        score += 10
                    if any(k in instruction for k in ["搜索", "查找", "输入", "填写", "查询", "评论", "发布", "发送"]):
                        if y <= 320:
                            score += 8
                score += keyword_hit_score(reason, ["返回", "关闭", "取消", "跳过", "搜索", "输入", "确认", "发送", "完成"])

            elif action == "TYPE":
                if text:
                    score += keyword_hit_score(text, ["搜索", "查找", "输入", "填写", "评论", "发布", "发送", "确认", "完成"])
                if input_flow:
                    score += 14
                if last_action == "TYPE":
                    score -= 2

            elif action == "OPEN":
                if self.state.step_count == 0:
                    score += 20
                if app_name and app_name in instruction:
                    score += 16

            elif action == "SCROLL":
                if any(k in instruction for k in ["滚动", "翻页", "浏览", "下滑", "上滑"]):
                    score += 16

            elif action == "COMPLETE":
                if any(k in instruction for k in ["完成", "结束", "提交", "done"]):
                    score += 18
                if last_action == "TYPE":
                    score += 8

            ranked.append((score, -idx, cand))

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return ranked[0][2] if ranked else {"action": "CLICK", "params": {"point": [500, 500]}, "source": "fallback", "score": 0, "reason": "empty candidate list"}

    def _finalize_candidate(self, candidate: dict, normalized_map: Dict[int, Any], instruction: str, history_actions: list) -> Tuple[str, Dict[str, Any], str, str]:
        """最终输出阶段：把选中的候选动作转成标准格式。"""
        action = candidate.get("action", "CLICK")
        params = candidate.get("params", {}) or {}
        raw_output = candidate.get("raw_output", "")
        expected_effect = candidate.get("expected_effect", "") or "画面发生变化"

        if action == "ENTER":
            action = "CLICK"
            params = {"point": self._top_candidate(normalized_map) or self._pick_click_target(normalized_map, prefer_top=True) or [500, 500]}
            expected_effect = "点击确认/搜索按钮"

        if action == "TYPE" and "text" in params:
            params = dict(params)
            params["text"] = self._dynamic_clean_type_text(params["text"])

        if action == "CLICK" and "point" in params:
            prefer_top = self._looks_like_input_flow(instruction, history_actions)
            params["point"] = self._normalize_click_point(params["point"], normalized_map, prefer_top=prefer_top)

            if prefer_top:
                top_point = self._top_candidate(normalized_map)
                if top_point and params["point"][1] > 260:
                    logger.info(f"🧲 输入流点击纠偏: {params['point']} -> {top_point}")
                    params["point"] = top_point

            # 仅在接近某个框时做小幅投影，避免跨区域硬拉
            best_point = self._best_bbox_target(params["point"], normalized_map, prefer_top=prefer_top)
            if best_point:
                dist = self._bbox_distance(params["point"], [best_point[0], best_point[1], best_point[0], best_point[1]])
                if dist < 80 and best_point != params["point"]:
                    logger.info(f"📦 局部bbox纠偏: {params['point']} -> {best_point}")
                    params["point"] = best_point

            params["point"] = self._clip_norm_point(params["point"])

        return action, params, expected_effect, raw_output

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Limit reached")

        img, element_map = draw_som_labels(input_data.current_image)
        original_width, original_height = input_data.current_image.size
        self._current_image_size = (original_width, original_height)

        normalized_map = {}
        for idx, meta in element_map.items():
            center = self._extract_center(meta)
            bbox = self._extract_bbox(meta)
            if center:
                normalized_meta = {"center": [int((center[0] / original_width) * 1000), int((center[1] / original_height) * 1000)]}
                if bbox:
                    normalized_meta["bbox"] = [
                        int((bbox[0] / original_width) * 1000),
                        int((bbox[1] / original_height) * 1000),
                        int((bbox[2] / original_width) * 1000),
                        int((bbox[3] / original_height) * 1000),
                    ]
                normalized_map[idx] = normalized_meta

        text_hints = extract_text_hints(input_data.current_image)
        text_hint_summary = summarize_text_hints(text_hints)

        img = draw_previous_action(img, input_data.history_actions)
        img = add_coordinate_grid(img)

        prompt = self._build_cot_prompt(input_data.instruction, input_data.history_actions, text_hint_summary)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                                 {"type": "image_url", "image_url": {"url": self._encode_image(img)}}]}]

        raw_output = ""
        usage = None
        try:
            temp = 0.4 if self.state.stuck_level >= 2 else 0.0
            response = self._call_api(messages, temperature=temp)
            raw_output = response.choices[0].message.content
            usage = self.extract_usage_info(response)
            model_action, model_params, model_effect = self._parse_with_effect(raw_output)
        except Exception as e:
            logger.warning(f"Model failed: {e}")
            model_action, model_params, model_effect = "CLICK", {"point": [500, 500]}, "兜底点击"
            raw_output = f"Fallback: {e}"

        candidates = self._build_candidates(
            input_data.instruction,
            input_data.history_actions,
            normalized_map,
            text_hints,
            model_action,
            model_params,
            model_effect,
            raw_output,
        )
        chosen = self._rank_candidates(candidates, input_data.instruction, input_data.history_actions)
        action, params, expected_effect, raw_output = self._finalize_candidate(
            chosen,
            normalized_map,
            input_data.instruction,
            input_data.history_actions,
        )

        action, params, normalized_effect = self._normalize_output(action, params, normalized_map, input_data.instruction)
        if normalized_effect:
            expected_effect = normalized_effect

        action, params = sanitize_and_stick(action, params, normalized_map)
        self.state.update(f"{action}:{params}", expected_effect)

        return AgentOutput(action=action, parameters=params, raw_output=raw_output, usage=usage)

    def _normalize_output(self, action, params, element_map, instruction):
        if action == "ENTER":
            top_point = self._top_candidate(element_map)
            if top_point:
                return "CLICK", {"point": top_point}, "点击可见确认/搜索按钮"
            return "CLICK", {"point": self._pick_click_target(element_map, prefer_top=True) or [500, 500]}, "确认输入并继续"
        if action not in self.VALID_ACTIONS:
            return self._generic_fallback_action(instruction, element_map)
        return action, params, ""

    def _generic_fallback_action(self, instruction, element_map):
        # 保持原有简单逻辑，但尽量避免屏幕中心盲点
        top_point = self._top_candidate(element_map)
        if top_point and any(k in instruction for k in ["搜索", "查找", "输入", "填写", "查询", "打车", "评论", "播放", "收藏", "关注"]):
            return "CLICK", {"point": top_point}, "聚焦输入框/搜索入口"
        return "CLICK", {"point": self._pick_click_target(element_map, prefer_top=False) or [500, 500]}, "兜底点击"
