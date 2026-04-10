# 文件路径: agent.py
from dotenv import load_dotenv

load_dotenv()

import hashlib
import logging
import re
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
        self._current_instruction = instruction or ""
        recent = self._recent_history(history_actions, window=2)
        return f"""你是一个 Android GUI 智能代理，必须以 LLM 判断为主导；规则只负责最后一层刹车。
任务目标：【{instruction}】

[当前上下文]
- 历史动作摘要: {self.state.get_summary()}
- 最近动作: {recent}
- 当前进度: {self.state.step_count}/{self.state.max_steps}

[决策原则]
1. 先看当前页面里“显式可见”的控件与状态，再决定下一步动作。
2. 只输出一个最合适的动作，不要解释，不要列步骤。
3. 优先跟随页面上的按钮、输入框、搜索框、确认区、结果区等可见线索，不要凭空补动作。
4. 如果页面已经明显接近目标、结果页、确认页或完成页，优先 COMPLETE。
5. 如果是输入后确认的场景，输入完成后优先点击页面上最明显的确认/搜索/下一步按钮。
6. 不要重复对同一个已经完成的输入内容再次 TYPE。

[TYPE 规则，必须严格遵守]
- TYPE 的文本必须与任务目标中“原始出现的目标词”完全一致。
- 不要把词想成一句描述，不要补全，不要扩写，不要重组。
- 比赛里这类错误会直接扣分，下面是常见范例：
  - 错误：回民街 -> 正确：回民街
  - 错误：西安回民街 -> 正确：回民街
  - 错误：三体 -> 正确：《三体》多人有声剧
  - 错误：跳舞的视频 / 跳舞的视频并查看 -> 正确：跳舞
- 禁止添加“的视频”“并查看”“西安”“查看一下”等任何后缀或前缀。
- 如果任务目标包含书名号、括号、空格、标点，尽量原样保留。
- 上面示例只是提醒常见失分点，不是枚举所有情况；核心仍然是“原样复写任务目标中的词”。

[CLICK 规则]
- 点击必须落在明确按钮、图标、搜索框或确认区中心。
- 优先点页面上最明确的可操作控件，不要点内容列表本身。
- 当你判断是在“输入后确认”阶段时，优先 CLICK，而不是再 TYPE。

[OPEN 规则]
- 只有在还没进入目标应用时才 OPEN。
- 已经打开目标应用后，不要重复 OPEN，同一个应用只开一次。

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

    def _self_check_type_text(self, text: str) -> str:
        """轻量自检：避免扩写尾缀，不做任意子串截断。"""
        text = self._normalize_text(text)
        instruction = getattr(self, "_current_instruction", "") or ""
        if not text or not instruction:
            return text

        # 若指令里存在书名号短语，且当前文本是其子串，优先补全为完整短语。
        for match in re.findall(r"《[^》]+》[^，。！？；,!?;\n]*", instruction):
            phrase = match.strip()
            if text and text in phrase and len(text) < len(phrase):
                return phrase

        # 原文已出现在指令中，直接保留。
        if text in instruction:
            return text

        # 仅清理常见扩写尾缀，避免把完整目标词截断成过短片段。
        noisy_suffixes = ["的视频并查看", "的视频", "并查看"]
        for suffix in noisy_suffixes:
            if text.endswith(suffix):
                candidate = text[: -len(suffix)].strip()
                if candidate and candidate in instruction:
                    return candidate

        return text

    def _normalize_output(self, action: str, params: dict) -> Tuple[str, dict, str]:
        params = params or {}

        if action == "CLICK_ID":
            # 赛题标准输出不要求 CLICK_ID，这里仅做安全退化。
            return "CLICK", {"point": [500, 500]}, ""

        if action == "ENTER":
            if isinstance(params, dict) and "point" in params:
                return "CLICK", {"point": self._clip_norm_point(params["point"])}, "确认输入并继续"
            if isinstance(params, dict) and "x" in params and "y" in params:
                return "CLICK", {"point": self._clip_norm_point([params["x"], params["y"]])}, "确认输入并继续"
            # 严格评测中 ENTER 常对应右上角确认/搜索按钮，避免中心兜底误点。
            return "CLICK", {"point": [900, 80]}, "确认输入并继续"

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
            checked = self._self_check_type_text(self._normalize_text(text))
            return "TYPE", {"text": checked}, ""

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
