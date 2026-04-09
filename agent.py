# 文件路径: agent.py
from dotenv import load_dotenv
load_dotenv()

import logging
import re
from typing import Dict, Any, Tuple
from agent_base import BaseAgent, AgentOutput, AgentInput

# 导入你自行封装的四大视觉与后处理工具
from utils.vision_enhancer import add_coordinate_grid
from utils.visual_memory import draw_previous_action
from utils.ui_detector import draw_som_labels
from utils.action_sandbox import sanitize_and_stick

# 导入我们在上一步构建的轻量级管理工具
from utils.state import GUIState
from utils.parser import robust_parse

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """通用 GUI Agent：模型优先，失败时使用通用兜底策略。"""

    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE", "ENTER"}

    def _initialize(self):
        """初始化全局状态"""
        self.state = GUIState(max_steps=45)

    def reset(self):
        """TestRunner 每次运行新用例时调用，重置状态机"""
        self.state = GUIState(max_steps=45)

    def _build_cot_prompt(self, instruction: str) -> str:
            history_summary = self.state.get_summary()
            plan_context = (
                "这是任务的第一步。先判断要打开哪个应用，再决定是点击、输入还是滚动。"
                if self.state.step_count == 0
                else f"【全局计划宏图】:\n{self.state.global_plan}\n请根据当前进度继续执行。"
            )
            stuck_warning = ""
            if self.state.stuck_level == 1:
                stuck_warning = "\n[提示] 如果当前页面没有进展，先尝试滚动寻找目标。"
            elif self.state.stuck_level >= 2:
                stuck_warning = "\n[紧急警告] 任务已陷入死循环！必须立即尝试 SCROLL 或者 CLICK 返回！"

            plan_line = "[Plan] 1. 识别应用 2. 找到入口 3. 完成任务" if self.state.step_count == 0 else ""
            return f"""你是一个高级 Android GUI 智能代理。
    任务目标：【{instruction}】

    [当前上下文]
    1. 历史动作: {history_summary}
    2. 当前进度: {self.state.step_count}/{self.state.max_steps}
    3. 【视觉提示】截图叠加了红线网格和数字标签。有标签直接估算标签中心坐标，没标签用网格估算。
    4. {plan_context}{stuck_warning}

    [通用交互与动作原则]
    - 【输入文字三步曲】：必须先 CLICK 输入框 -> TYPE 输入文字 -> 必须 ENTER 提交！
    - 🛑 【绝对禁忌】：如果任务是“播放”、“收藏”、“查看”或“发布评论”，只要画面上【已经显示正在播放】、【按钮已亮起变色】或【内容已上屏显示】，这就是最终终点！绝对、绝对不允许再做任何点击或多余操作！直接交卷！

    请按以下结构输出，并且只输出一组动作决策：
    {plan_line}
    [Observation] 灵魂拷问：任务的最终目的是什么？当前的画面状态是否已经达成了这个目的？（例如：要求播放，现在正在播放页面吗？要求收藏，现在收藏图标亮了吗？）
    [Reflection] 如果Observation确认目标已达成，我必须停止一切操作！
    [Thought] 明确写出：“目标已达成，我要输出 COMPLETE” 或 “目标未达成，我需要继续操作”。
    [Action] 只允许输出以下六种之一：
    CLICK:[[x, y]]
    TYPE:['文本']
    ENTER:[]
    SCROLL:[[x1, y1], [x2, y2]]
    OPEN:['应用名']
    COMPLETE:[]
    [Expected Effect] 预期变化。
    """

    def _parse_with_effect(self, raw_text: str) -> Tuple[str, Dict[str, Any], str]:
        action_block = raw_text
        if "[Action]" in raw_text:
            action_block = raw_text.split("[Action]")[-1].split("[Expected Effect]")[0]
        elif "Action:" in raw_text:
            action_block = raw_text.split("Action:")[-1].split("[Expected Effect]")[0]

        action, params = robust_parse(action_block)
        expected_effect = "画面发生变化"
        if "[Expected Effect]" in raw_text:
            expected_effect = raw_text.split("[Expected Effect]")[-1].strip()
        return action, params, expected_effect

    def _infer_app_name(self, instruction: str) -> str:
        """从自然语言指令中尽量泛化地提取要打开的应用名。"""
        verbs = ["打开", "进入", "搜索", "查找", "查看", "播放", "打车", "发布", "评论", "收藏", "输入", "选择", "购买", "下载", "预约", "登录", "注册", "分享", "关注", "浏览", "点开", "下单", "确认"]
        text = instruction.strip()

        # 优先处理类似“去X打开Y”或“在X搜索Y”中的 X
        for prefix in ["去", "在", "到", "打开", "进入"]:
            if prefix in text:
                start = text.find(prefix) + len(prefix)
                remaining = text[start:]
                cut_positions = [remaining.find(v) for v in verbs if remaining.find(v) != -1]
                cut_positions = [p for p in cut_positions if p >= 0]
                if cut_positions:
                    candidate = remaining[:min(cut_positions)]
                else:
                    candidate = remaining
                candidate = re.sub(r"[，,。；;：:].*$", "", candidate).strip()
                candidate = candidate.strip(" 的里中上")
                if candidate:
                    return candidate

        # 兜底：抓取“应用/平台”附近的连续文本
        match = re.search(r"([\u4e00-\u9fa5A-Za-z0-9]{2,20})(?:搜索|查看|播放|评论|收藏|打车|发布|输入|选择|购买|下载|预约|登录|注册|分享|关注|浏览|点开|下单|确认)", text)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _copy_params(params: Dict[str, Any]) -> Dict[str, Any]:
        copied: Dict[str, Any] = {}
        for key, value in params.items():
            copied[key] = list(value) if isinstance(value, list) else value
        return copied

    @staticmethod
    def _pick_focus_point(element_map: Dict[int, Any]) -> Tuple[int, int]:
        if not element_map:
            return 500, 120
        candidates = sorted(
            [tuple(v) for v in element_map.values() if isinstance(v, (list, tuple)) and len(v) == 2],
            key=lambda p: (p[1], abs(p[0] - 500))
        )
        if not candidates:
            return 500, 120
        x, y = candidates[0]
        return int(x), int(y)

    def _generic_fallback_action(self, instruction: str, element_map: Dict[int, Any]) -> Tuple[str, Dict[str, Any], str]:
        app_name = self._infer_app_name(instruction)
        if self.state.step_count == 0 and app_name:
            return "OPEN", {"app_name": app_name}, f"打开{app_name}"

        if self.state.stuck_level >= 2:
            return "SCROLL", {"start_point": [500, 780], "end_point": [500, 260]}, "页面向下滚动寻找目标"

        if any(k in instruction for k in ["搜索", "查找", "输入", "填写", "查询"]):
            x, y = self._pick_focus_point(element_map)
            return "CLICK", {"point": [x, max(80, min(y, 220))]}, "聚焦输入框"

        if any(k in instruction for k in ["返回", "关闭", "取消", "跳过"]):
            return "CLICK", {"point": [60, 80]}, "关闭当前遮挡"

        if any(k in instruction for k in ["滚动", "下滑", "上滑", "翻页"]):
            return "SCROLL", {"start_point": [500, 760], "end_point": [500, 260]}, "页面滚动"

        x, y = self._pick_focus_point(element_map)
        if y < 300:
            return "CLICK", {"point": [x, y]}, "页面元素被点击"
        return "CLICK", {"point": [500, 500]}, "页面发生变化"

    def _normalize_output(self, action: str, params: Dict[str, Any], element_map: Dict[int, Any], instruction: str) -> Tuple[str, Dict[str, Any], str]:
        if action not in self.VALID_ACTIONS:
            return self._generic_fallback_action(instruction, element_map)
        if action == "COMPLETE":
            return "COMPLETE", {}, "任务完成"
        return action, self._copy_params(params), ""

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Force stop limit reached")

        img, element_map = draw_som_labels(input_data.current_image)
        original_width, original_height = input_data.current_image.size
        normalized_element_map = {}
        if element_map:
            for idx, center in element_map.items():
                if isinstance(center, (list, tuple)) and len(center) == 2:
                    # 将 x 和 y 分别映射到 0-1000
                    nx = int((center[0] / original_width) * 1000)
                    ny = int((center[1] / original_height) * 1000)
                    normalized_element_map[idx] = [nx, ny]

        # 将归一化后的 map 覆盖原本的 element_map，确保后续所有吸附逻辑(Sticky Click)都在同一个维度
        element_map = normalized_element_map
        img = draw_previous_action(img, input_data.history_actions)
        img = add_coordinate_grid(img)

        prompt = self._build_cot_prompt(input_data.instruction)
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(img)}}
            ]}
        ]

        raw_output = ""
        usage_info = None
        action = ""
        params: Dict[str, Any] = {}
        expected_effect = ""

        try:
            temp = 0.4 if self.state.stuck_level >= 2 else 0.0
            response = self._call_api(messages, temperature=temp)
            raw_output = response.choices[0].message.content
            usage_info = self.extract_usage_info(response)

            if self.state.step_count == 0 and "[Plan]" in raw_output:
                try:
                    plan_text = raw_output.split("[Plan]")[-1].split("[Observation]")[0].strip()
                    self.state.global_plan = plan_text if plan_text else input_data.instruction
                except Exception:
                    self.state.global_plan = input_data.instruction

            action, params, expected_effect = self._parse_with_effect(raw_output)
        except Exception as e:
            logger.warning(f"[Agent] Model path failed, using generic fallback: {e}")
            action, params, expected_effect = self._generic_fallback_action(input_data.instruction, element_map)
            raw_output = f"Fallback: {e}"

        action, params, normalized_effect = self._normalize_output(action, params, element_map, input_data.instruction)
        if normalized_effect:
            expected_effect = normalized_effect
        if not expected_effect:
            expected_effect = "画面发生变化"

        action, params = sanitize_and_stick(action, params, element_map)

        if self.state.stuck_level >= 3 and action != "COMPLETE":
            action = "CLICK"
            params = {"point": [50, 80]}

        self.state.update(f"{action}:{params}", expected_effect)

        return AgentOutput(
            action=action,
            parameters=params,
            raw_output=raw_output,
            usage=usage_info,
        )
