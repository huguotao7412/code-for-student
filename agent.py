# 文件路径: agent.py
from dotenv import load_dotenv

load_dotenv()
import io
import base64
from PIL import Image
import hashlib
import json
import logging
import re
from typing import Any, Dict, Tuple

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - fallback for environments without langgraph
    END = None
    START = None
    StateGraph = None

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.graph_state import WorkflowState
from utils.nodes import actor_node, format_output_node, planner_node, reviewer_node
from utils.parser import robust_parse
from utils.state import GUIState
from utils.vision_enhancer import add_coordinate_grid

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """LLM 主导，最小 schema 适配。"""

    # 【修复1】严格对齐官方 5 大动作，彻底铲除 ENTER
    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}

    def _initialize(self):
        self.state = GUIState(max_steps=45)
        self._plan_instruction = ""
        self._task_plan = []
        self._current_instruction = ""
        self.graph = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            logger.warning("langgraph is not available, fallback to legacy single-agent runtime")
            return None
        workflow = StateGraph(WorkflowState)

        workflow.add_node("planner", lambda state: planner_node(state, self))
        workflow.add_node("actor", lambda state: actor_node(state, self))
        workflow.add_node("reviewer", lambda state: reviewer_node(state, self))
        workflow.add_node("format_output", lambda state: format_output_node(state, self))

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "actor")
        workflow.add_edge("actor", "reviewer")

        def _review_route(state: WorkflowState) -> str:
            if state.get("reviewer_feedback") == "PASS" or int(state.get("retry_count", 0)) >= 2:
                return "format_output"
            return "actor"

        workflow.add_conditional_edges(
            "reviewer",
            _review_route,
            {
                "actor": "actor",
                "format_output": "format_output",
            },
        )
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def reset(self):
        self.state = GUIState(max_steps=45)
        self._plan_instruction = ""
        self._task_plan = []
        self._current_instruction = ""

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

    def _encode_image(self, image: Image.Image, image_format: str = "JPEG") -> str:
        try:
            enhanced_image = add_coordinate_grid(image)
        except Exception as e:
            logger.warning(f"网格渲染失败，降级使用原图: {e}")
            enhanced_image = image.copy()

        img = enhanced_image.convert("RGB")
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    def _build_plan_prompt(self, instruction: str) -> str:
        return f"""你是 GUI 任务规划器。请先理解任务，再输出 4~8 条结构化子步骤。
要求：
1) 仅输出 JSON，不要解释。
2) JSON 格式：
{{
  "sub_steps": [
    {{"id": 1, "stage": "阶段名", "goal": "要完成什么", "action_hint": "OPEN/CLICK/TYPE/SCROLL/COMPLETE"}}
  ]
}}
3) 计划要可迁移，不依赖固定坐标。

Few-shot 示例：
[示例1]
任务：去喜马拉雅搜索《三体》多人有声剧并播放
{{"sub_steps":[
{{"id":1,"stage":"打开应用","goal":"进入喜马拉雅","action_hint":"OPEN"}},
{{"id":2,"stage":"处理遮挡","goal":"关闭弹窗或引导层","action_hint":"CLICK"}},
{{"id":3,"stage":"进入搜索","goal":"点击搜索框或搜索入口","action_hint":"CLICK"}},
{{"id":4,"stage":"输入关键词","goal":"输入《三体》多人有声剧","action_hint":"TYPE"}},
{{"id":5,"stage":"确认搜索","goal":"执行搜索","action_hint":"CLICK"}},
{{"id":6,"stage":"选择结果","goal":"进入目标音频","action_hint":"CLICK"}},
{{"id":7,"stage":"播放","goal":"点击播放控件","action_hint":"CLICK"}},
{{"id":8,"stage":"结束","goal":"任务完成","action_hint":"COMPLETE"}}
]}}

[示例2]
任务：去爱奇艺搜索电视剧并发表评论
{{"sub_steps":[
{{"id":1,"stage":"打开应用","goal":"进入爱奇艺","action_hint":"OPEN"}},
{{"id":2,"stage":"处理遮挡","goal":"关闭弹窗","action_hint":"CLICK"}},
{{"id":3,"stage":"进入搜索","goal":"点击搜索入口","action_hint":"CLICK"}},
{{"id":4,"stage":"进入剧集","goal":"搜索并进入目标剧集页面","action_hint":"CLICK"}},
{{"id":5,"stage":"进入评论","goal":"打开评论区","action_hint":"CLICK"}},
{{"id":6,"stage":"发布评论","goal":"输入并发送评论","action_hint":"TYPE"}},
{{"id":7,"stage":"结束","goal":"任务完成","action_hint":"COMPLETE"}}
]}}

[示例3]
任务：去哔哩哔哩搜索视频并查看
{{"sub_steps":[
{{"id":1,"stage":"打开应用","goal":"进入哔哩哔哩","action_hint":"OPEN"}},
{{"id":2,"stage":"进入搜索","goal":"点击搜索框","action_hint":"CLICK"}},
{{"id":3,"stage":"输入关键词","goal":"输入目标关键词","action_hint":"TYPE"}},
{{"id":4,"stage":"确认搜索","goal":"点击搜索按钮","action_hint":"CLICK"}},
{{"id":5,"stage":"选择结果","goal":"进入目标视频","action_hint":"CLICK"}},
{{"id":6,"stage":"结束","goal":"任务完成","action_hint":"COMPLETE"}}
]}}

现在请只针对下面任务生成计划：
任务：{instruction}
"""

    def _plan_to_text(self) -> str:
        if not self._task_plan:
            return "无"
        lines = []
        for item in self._task_plan:
            step_id = item.get("id", "?")
            stage = item.get("stage", "")
            goal = item.get("goal", "")
            action_hint = item.get("action_hint", "")
            lines.append(f"{step_id}. {stage} | {goal} | hint={action_hint}")
        return "\\n".join(lines)

    def _parse_plan(self, raw_text: str) -> list:
        try:
            data = json.loads(raw_text)
            sub_steps = data.get("sub_steps", []) if isinstance(data, dict) else []
            parsed = []
            for i, item in enumerate(sub_steps, start=1):
                if not isinstance(item, dict):
                    continue
                parsed.append({
                    "id": int(item.get("id", i)),
                    "stage": str(item.get("stage", "")).strip(),
                    "goal": str(item.get("goal", "")).strip(),
                    "action_hint": str(item.get("action_hint", "")).strip().upper(),
                })
            if 4 <= len(parsed) <= 8:
                return parsed
        except Exception:
            pass
        return [
            {"id": 1, "stage": "打开或进入目标应用", "goal": "到达任务主场景", "action_hint": "OPEN"},
            {"id": 2, "stage": "定位入口", "goal": "找到搜索/功能入口", "action_hint": "CLICK"},
            {"id": 3, "stage": "输入", "goal": "输入任务关键词", "action_hint": "TYPE"},
            {"id": 4, "stage": "确认", "goal": "确认搜索或下一步", "action_hint": "CLICK"},
            {"id": 5, "stage": "结果处理", "goal": "进入目标结果页面", "action_hint": "CLICK"},
            {"id": 6, "stage": "完成", "goal": "任务结束", "action_hint": "COMPLETE"},
        ]

    def _ensure_task_plan(self, instruction: str, current_image) -> None:
        if self._plan_instruction == instruction and self._task_plan:
            return

        self._plan_instruction = instruction
        self._task_plan = []
        plan_prompt = self._build_plan_prompt(instruction)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": plan_prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(current_image)}},
            ],
        }]
        try:
            response = self._call_api(messages)
            raw_plan = response.choices[0].message.content
            self._task_plan = self._parse_plan(raw_plan)
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            self._task_plan = self._parse_plan("")

    def _scenario_hints(self, instruction: str) -> str:
        text = instruction or ""
        hints = []
        if any(k in text for k in ["地图", "打车", "导航", "去", "前往", "路线", "目的地", "起点", "终点"]):
            hints.append(
                "- 地图/导航类：起终点输入后常需点候选项确认，确认后再进入下一字段。特别注意：未见终点输入框 caret，不得直接 TYPE 终点词。")
        if any(k in text for k in ["播放", "听", "收听", "观看", "视频", "剧集", "电影", "有声"]):
            hints.append(
                "- 播放类：常见链路为 搜索 -> 选择结果 -> 触发播放。优先点击可见的播放控件，不要把单纯的标题文本当作播放入口。")
        if any(k in text for k in ["买", "外卖", "商品", "店铺", "预订", "定一个", "酒店"]):
            hints.append("- 购物/O2O类：注意处理各种促销或授权弹窗。优先寻找“加购物车”、“预订”或“结算”按钮。")
        if any(k in text for k in ["搜索", "搜", "查找"]):
            hints.append("- 搜索类通用：输入后务必 CLICK 原页面上方或右侧的原生搜索按钮！")
        return "\n".join(hints) if hints else "- 通用策略：优先跟随当前页面可见的 UI 控件证据，遇弹窗先关闭。"

    def _build_prompt(
            self,
            instruction: str,
            history_actions: list,
            reviewer_feedback: str = "",
            retry_count: int = 0,
    ) -> str:
        self._current_instruction = instruction or ""
        recent = self._recent_history(history_actions, window=2)
        review_section = ""
        if reviewer_feedback and reviewer_feedback != "PASS":
            review_section = f"""
[Reviewer 反馈]
- 当前重试次数: {retry_count}
- 审核意见: {reviewer_feedback}
- 🔴 你必须先修复审核意见，再给下一步动作！
"""
        return f"""你是 Android GUI 智能代理。每步必须按 ReAct：观察 -> 分析 -> 动作。
任务目标：【{instruction}】

[当前状态]
- 历史摘要: {self.state.get_summary()}
- 最近动作: {recent}
- 步数: {self.state.step_count}/{self.state.max_steps}
- 当前任务计划(4~8步):
{self._plan_to_text()}
{review_section}

[状态机声明（强制）]
在分析之前，你必须严格评估并声明当前所处的任务状态，只能从以下状态库中选择一个：
- [State: 寻找入口]：目标是找搜索框/分类等起始入口。
- [State: 处理弹窗]：目标是关掉广告/权限确认/升级提示等遮挡物。
- [State: 文本输入]：目标是打字（必须确认光标已在输入框内）。
- [State: 确认搜索]：目标是点击页面上的原生搜索按钮确认输入。
- [State: 结果筛选]：目标是在列表中浏览并点击正确结果。
- [State: 终态验证]：目标结果已呈现，准备结束任务。

[合法动作] (注意：严禁输出 ENTER 动作)
CLICK / TYPE / SCROLL / OPEN / COMPLETE

[最高优先级规则]
1) 键盘禁区警告（致命硬约束）：本评测中，屏幕下方弹出的【系统虚拟键盘】属于非法点击区域！绝对禁止 CLICK 键盘上的任何区域（包括键盘上的搜索/回车键）。确认搜索必须 CLICK 页面内容区（通常在输入框右侧或联想词列表）的原生控件！
2) 弹窗遮挡最高优先级：若存在广告或权限弹窗，必须先关掉再做其他动作。
3) 输入框激活悖论：搜索框通常需点两次！第一次点击激活（出现光标 caret），此时才允许 TYPE。
4) 搜索框状态规则：只要没有可见光标 caret，就一律视为“未激活”，先 CLICK 聚焦。若已激活，禁止重复 CLICK。
5) 地图双输入框：起点确认后，必须先进入终点输入入口再 TYPE 终点。
6) 输入后确认规则：刚 TYPE 后，禁止盲目连续 TYPE，必须 CLICK 页面原生 UI 里的“搜索/确认”按键。

[COMPLETE 触发规则]
- 仅当状态为 [State: 终态验证] 且目标结果实质达成时，才允许 COMPLETE。

[文本输入规则]
- 提取文本时请自行判断是否需要包含“附近”或城市前缀。保证语义完整。
- 示例（地图类）：错误：西安回民街 -> 正确：回民街

[UI 场景类别经验]
{self._scenario_hints(instruction)}

[输出格式]
请在最后一行输出最终动作。禁止在 [Analyze] 中使用坐标 [[x, y]] 以免干扰解析！
必须严格按以下结构输出：
[State] [State: 选择上述六个状态之一]
[Observe] 只描述当前屏幕可见事实
[Analyze] 结合当前状态，判断下一步意图
[Action] 仅一条动作，格式如下之一：
CLICK:[[x,y]]
TYPE:['文本']
SCROLL:[[x1,y1],[x2,y2]]
OPEN:['应用名']
COMPLETE:[]
[Expected Effect] 简述执行后应看到的变化
"""

    @staticmethod
    def _parse_with_effect(raw_text: str) -> Tuple[str, Dict[str, Any], str]:
        action_block = raw_text
        if "[Action]" in raw_text:
            action_block = raw_text.split("[Action]")[-1].split("[Expected Effect]")[0]
        action, params = robust_parse(action_block)
        expected_effect = raw_text.split("[Expected Effect]")[
            -1].strip() if "[Expected Effect]" in raw_text else "画面发生变化"
        return action, params, expected_effect

    @staticmethod
    def _clip_norm_point(point: list) -> list:
        if not point or len(point) != 2:
            return [500, 500]
        return [max(10, min(990, int(point[0]))), max(10, min(990, int(point[1])))]

    @staticmethod
    def _normalize_text(text: Any) -> str:
        raw = "" if text is None else str(text)
        return raw.strip().strip("'\" ")

    def _self_check_type_text(self, text: str) -> str:
        text = self._normalize_text(text)
        instruction = getattr(self, "_current_instruction", "") or ""
        if not text or not instruction:
            return text
        for match in re.findall(r"《[^》]+》[^，。！？；,!?;\n]*", instruction):
            phrase = match.strip()
            if text and text in phrase and len(text) < len(phrase):
                return phrase
        if text in instruction:
            return text
        return text

    def _normalize_output(self, action: str, params: dict) -> Tuple[str, dict, str]:
        params = params or {}

        if action == "CLICK_ID":
            return "CLICK", {"point": [500, 500]}, ""

        # 【修复2】去掉了危险的 ENTER 强行映射 [900, 80] 的逻辑

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
            # 【修复3】同时返回 content 和 text，完美兼容官方解析器
            return "TYPE", {"content": checked, "text": checked}, ""

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

    def _legacy_act(self, input_data: AgentInput) -> AgentOutput:
        current_signature = self._image_signature(input_data.current_image)

        if input_data.step_count <= 1 or self._plan_instruction != input_data.instruction:
            self._ensure_task_plan(input_data.instruction, input_data.current_image)

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

        last_action = ""
        if input_data.history_actions:
            last = input_data.history_actions[-1]
            if isinstance(last, dict):
                last_action = str(last.get("action", "")).upper()

        # 【修复4】去掉了危险的 legacy 连续 TYPE 回退 [900, 80] 的逻辑，交给 Reviewer 阻断

        if model_effect and not expected_effect:
            expected_effect = model_effect

        self.state.update(
            f"{action}:{params}",
            expected_effect or model_effect or "画面发生变化",
            visual_hash=current_signature,
        )
        self.state.last_image = input_data.current_image.copy()

        return AgentOutput(action=action, parameters=params, raw_output=raw_output, usage=usage)

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Limit reached")

        if not self.graph:
            return self._legacy_act(input_data)

        current_signature = self._image_signature(input_data.current_image)
        initial_state: WorkflowState = {
            "input_data": input_data,
            "plan_instruction": self._plan_instruction,
            "task_plan": self._task_plan,
            "history_actions": input_data.history_actions,
            "proposed_action": "",
            "proposed_params": {},
            "model_effect": "",
            "reviewer_feedback": "",
            "retry_count": 0,
            "raw_output": "",
            "usage": None,
        }

        try:
            final_state = self.graph.invoke(initial_state)
            output = final_state.get("final_output")
            if not isinstance(output, AgentOutput):
                output = AgentOutput(
                    action=final_state.get("normalized_action", "CLICK"),
                    parameters=final_state.get("normalized_params", {"point": [500, 500]}),
                    raw_output=final_state.get("raw_output", ""),
                    usage=final_state.get("usage"),
                )
            expected_effect = final_state.get("expected_effect", "画面发生变化")
        except Exception as e:
            logger.warning(f"Graph invoke failed, fallback to legacy: {e}")
            return self._legacy_act(input_data)

        self.state.update(
            f"{output.action}:{output.parameters}",
            expected_effect or "画面发生变化",
            visual_hash=current_signature,
        )
        self.state.last_image = input_data.current_image.copy()

        return output