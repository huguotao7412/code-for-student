# 文件路径: agent.py
from dotenv import load_dotenv

load_dotenv()

import hashlib
import logging
from typing import Any, Dict, Tuple

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.parser import robust_parse
from utils.state import GUIState

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """LLM 主导，最小 schema 适配。"""

    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}

    def _initialize(self):
        self.state = GUIState(max_steps=45)

    def reset(self):
        self.state = GUIState(max_steps=45)

    @staticmethod
    def _recent_history(history_actions: list, window: int = 2) -> list:
        if not history_actions or window <= 0:
            return []
        return history_actions[-window:]

    def _image_signature(self, image) -> str:
        try:
            thumb = image.convert("L").resize((32, 32))
            return hashlib.sha1(thumb.tobytes()).hexdigest()
        except Exception:
            return ""

    def _build_prompt(self, instruction: str, history_actions: list) -> str:
        recent = self._recent_history(history_actions, window=2)
        return f"""你是一个 Android GUI 智能代理。
任务目标：【{instruction}】

[当前上下文]
- 历史动作摘要: {self.state.get_summary()}
- 最近动作: {recent}
- 当前进度: {self.state.step_count}/{self.state.max_steps}

[强约束]
- 搜索/查找/输入/填写/查询 场景优先使用 TYPE 或必要的点击聚焦；不要把搜索任务误当成 OPEN。
- TYPE 必须完整复写目标文本，尤其是书名号、括号、标点、空格都尽量原样保留。
- 如果已经进入目标应用，禁止再次 OPEN 同一个应用；应直接在当前界面继续操作。
- 只输出一个动作，不要解释。

[可用动作]
CLICK:[[x,y]]
TYPE:['文本']
SCROLL:[[x1,y1],[x2,y2]]
OPEN:['应用名']
COMPLETE:[]
"""

    @staticmethod
    def _parse_with_effect(raw_text: str) -> Tuple[str, Dict[str, Any], str]:
        action_block = raw_text
        if "[Action]" in raw_text:
            action_block = raw_text.split("[Action]")[-1].split("[Expected Effect]")[0]
        action, params = robust_parse(action_block)
        expected_effect = raw_text.split("[Expected Effect]")[-1].strip() if "[Expected Effect]" in raw_text else "画面发生变化"
        return action, params, expected_effect

    @staticmethod
    def _clip_norm_point(point: list) -> list:
        if not point or len(point) != 2:
            return [500, 500]
        return [max(10, min(990, int(point[0]))), max(10, min(990, int(point[1])))]

    @staticmethod
    def _normalize_text(text: Any) -> str:
        raw = "" if text is None else str(text)
        # 只去掉模型常见外层包装，不碰正文。
        return raw.strip().strip("'\" ")

    def _normalize_output(self, action: str, params: dict) -> Tuple[str, dict, str]:
        params = params or {}

        if action == "CLICK_ID":
            # 赛题标准输出不要求 CLICK_ID，这里仅做安全退化。
            return "CLICK", {"point": [500, 500]}, ""

        if action == "ENTER":
            return "CLICK", {"point": [500, 500]}, "确认输入并继续"

        if action == "CLICK":
            if isinstance(params, dict) and "point" in params:
                return "CLICK", {"point": self._clip_norm_point(params["point"])}, ""
            if isinstance(params, dict) and "x" in params and "y" in params:
                return "CLICK", {"point": self._clip_norm_point([params["x"], params["y"]])}, ""
            return "CLICK", {"point": [500, 500]}, ""

        if action == "TYPE":
            text = ""
            if isinstance(params, dict):
                text = params.get("text", params.get("content", ""))
            return "TYPE", {"text": self._normalize_text(text)}, ""

        if action == "SCROLL":
            if isinstance(params, dict):
                start = params.get("start_point")
                end = params.get("end_point")
                points = params.get("points")
                if (not start or not end) and isinstance(points, list) and len(points) == 2:
                    start, end = points[0], points[1]
                if start and end:
                    return "SCROLL", {
                        "start_point": self._clip_norm_point(start),
                        "end_point": self._clip_norm_point(end),
                    }, ""
            return "SCROLL", {"start_point": [500, 800], "end_point": [500, 200]}, ""

        if action == "OPEN":
            app_name = ""
            if isinstance(params, dict):
                app_name = params.get("app_name", params.get("app", params.get("content", "")))
            return "OPEN", {"app_name": self._normalize_text(app_name)}, ""

        if action == "COMPLETE":
            return "COMPLETE", {}, ""

        if action not in self.VALID_ACTIONS:
            return "CLICK", {"point": [500, 500]}, "兜底点击"

        return action, params, ""

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Limit reached")

        current_signature = self._image_signature(input_data.current_image)
        prompt = self._build_prompt(input_data.instruction, input_data.history_actions)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(input_data.current_image)}},
            ],
        }]

        raw_output = ""
        usage = None
        try:
            response = self._call_api(messages, temperature=0.0)
            raw_output = response.choices[0].message.content
            usage = self.extract_usage_info(response)
            model_action, model_params, model_effect = self._parse_with_effect(raw_output)
        except Exception as e:
            logger.warning(f"Model failed: {e}")
            model_action, model_params, model_effect = "CLICK", {"point": [500, 500]}, "兜底点击"
            raw_output = f"Fallback: {e}"

        action, params, expected_effect = self._normalize_output(model_action, model_params)
        if model_effect and not expected_effect:
            expected_effect = model_effect

        self.state.update(
            f"{action}:{params}",
            expected_effect or model_effect or "画面发生变化",
            visual_hash=current_signature,
        )
        self.state.last_image = input_data.current_image.copy()

        return AgentOutput(action=action, parameters=params, raw_output=raw_output, usage=usage)
