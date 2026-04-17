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
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
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
    """
    depth research agent
    GUI 智能代理 - 基于 A2A 协议与并发 Actor 架构
    """

    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}

    def _initialize(self):
        self.state = GUIState(max_steps=45)
        self._plan_instruction = ""
        self._task_plan = []
        self._app_name = ""
        self._current_instruction = ""
        self.graph = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            logger.warning("langgraph 不可用，回退到基础模式")
            return None
        workflow = StateGraph(WorkflowState)

        workflow.add_node("planner", lambda state: planner_node(state, self))
        # Actor 节点现在负责管理并发调度
        workflow.add_node("actor", lambda state: self._concurrent_actor_executor(state))
        workflow.add_node("reviewer", lambda state: reviewer_node(state, self))
        workflow.add_node("format_output", lambda state: format_output_node(state, self))

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "actor")
        workflow.add_edge("actor", "reviewer")

        def _review_route(state: WorkflowState) -> str:
            decision = state.get("reviewer_decision", "")
            if decision == "PASS" or state.get("reviewer_feedback") == "PASS" or int(state.get("retry_count", 0)) >= 1:
                return "format_output"
            return "actor"

        workflow.add_conditional_edges("reviewer", _review_route, {"actor": "actor", "format_output": "format_output"})
        workflow.add_edge("format_output", END)

        return workflow.compile()

    def _concurrent_actor_executor(self, state: WorkflowState) -> Dict[str, Any]:
        """
        根据 Planner 规划的多个 Actor 实例，进行并发的界面分析与决策。
        """
        concurrent_actors = state.get("concurrent_actors", [])
        if not concurrent_actors:
            # 回退到标准单 Actor
            return actor_node(state, self)

        logger.info(f"[Actor Executor] 启动并发 Actor 池，数量: {len(concurrent_actors)}")

        results = []
        # 使用并发执行器同时拉起多个 Actor 实例分析当前截图
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(concurrent_actors)) as executor:
            # 为了区分不同 Actor 的角色，可以在状态中临时注入角色描述
            future_to_actor = {}
            for actor_cfg in concurrent_actors:
                # 浅拷贝状态并注入专属配置
                actor_state = dict(state)
                actor_state["current_actor_role"] = actor_cfg.get("role", "")
                future = executor.submit(actor_node, actor_state, self)
                future_to_actor[future] = actor_cfg

            for future in concurrent.futures.as_completed(future_to_actor):
                try:
                    res = future.result()
                    if res and res.get("proposed_action") != "CLICK" or res.get("proposed_params") != {
                        "point": [500, 500]}:
                        results.append(res)
                except Exception as e:
                    logger.error(f"[Actor Executor] 并发 Actor 发生异常: {e}")

        # 优先选择非兜底的有效动作，交由 Reviewer 裁决
        if results:
            best_result = results[0]  # 这里可以加入更复杂的选举逻辑
            logger.info(f"[Actor Executor] 并发分析完毕，采纳动作: {best_result.get('proposed_action')}")
            return best_result

        return actor_node(state, self)

    def reset(self):
        self.state = GUIState(max_steps=45)
        self._plan_instruction = ""
        self._task_plan = []
        self._app_name = ""
        self._current_instruction = ""

    def _ensure_task_plan(self, instruction: str, current_image: Image.Image):
        """
        调用大模型进行全局任务拆解，提取 App 名称与任务计划。
        """
        system_prompt = (
            "你是一个 GUI 智能代理架构中的全局规划师(Planner)。请仔细分析用户的手机操作指令。\n"
            "【App名称映射与纠错规则】（最高优先级）：\n"
            " 1. 美团专项：主办方环境没有独立的‘美团外卖’APP。如果任务涉及点外卖，必须将 'app_name' 识别为‘美团’，并在 'sub_tasks' 的第一步明确设置为‘在美团首页寻找并点击外卖入口’。\n"
            " 2. 名称纠错：如果用户指令中出现“去哪旅行”这个APP名称，必须强制纠正，将 'app_name' 严格输出为‘去哪儿旅行’（绝对不能漏掉“儿”字）。\n\n"
            "你需要输出一个严谨的 JSON 格式结果，包含以下两个字段：\n"
            "1. 'app_name': 用户指令中要求打开的目标 App 名称（如无明确说明，请根据常识推断）。\n"
            "2. 'sub_tasks': 将该用户的总指令分解为 6 到 8 个按先后顺序排列的小任务列表。\n"
            "   每个子任务必须是一个包含以下字段的字典：\n"
            "   - 'id': 任务的数字序号 (1, 2, 3...)\n"
            "   - 'stage': 任务阶段简述 (如 '进入应用', '搜索目标', '确认输入')\n"
            "   - 'goal': 详细的任务目标描述\n"
            "   - 'action_hint': 建议的动作类型提示 (如 'OPEN', 'TYPE', 'CLICK')\n"
            "【特殊规划规则】：\n"
            " - 如果是[出行/打车/买票/导航]类任务，必须明确拆分出独立的子任务：先“找起点”，再“找终点”。如果指令中包含出发时间（如明天、后天、特定日期），必须在“找终点”之后单独设置一个“选择日期”的子任务。\n"
            "【严格要求】：只输出合法的 JSON 字符串，绝对不要包含 markdown 标记 (如 ```json)、多余的解释或前后缀文本。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户指令：{instruction}"}
        ]

        try:
            # 规划阶段不需要视觉图，仅依赖纯文本推理，温度设低保证 JSON 格式稳定性
            response = self._call_api(messages, temperature=0.1)
            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            result = json.loads(content)
            self._app_name = result.get("app_name", "未知应用")
            self._task_plan = result.get("sub_tasks", [])
            self._plan_instruction = instruction
            logger.info(f"[Planner] 成功生成计划，包含 {len(self._task_plan)} 个子任务。")

        except Exception as e:
            logger.error(f"[Planner] 任务拆解或解析失败: {e}，启用基础兜底计划")
            self._app_name = "未知应用"
            self._plan_instruction = instruction
            self._task_plan = [
                {"id": 1, "stage": "探索启动", "goal": "在桌面上寻找并打开目标应用", "action_hint": "OPEN"},
                {"id": 2, "stage": "界面加载", "goal": "确认应用的主界面已加载完成", "action_hint": "NONE"},
                {"id": 3, "stage": "寻找入口", "goal": "寻找并点击搜索框或相关的核心入口", "action_hint": "CLICK"},
                {"id": 4, "stage": "输入信息", "goal": "在激活的输入框中键入目标关键词", "action_hint": "TYPE"},
                {"id": 5, "stage": "确认提交", "goal": "寻找并点击键盘或原生的搜索/确认按钮", "action_hint": "CLICK"},
                {"id": 6, "stage": "验证完成", "goal": "确认目标画面出现，结束任务", "action_hint": "COMPLETE"}
            ]

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

    def _focus_signature(self, image) -> str:
        try:
            gray = image.convert("L")
            w, h = gray.size
            top = gray.crop((0, 0, w, max(1, int(h * 0.34))))
            bottom = gray.crop((0, int(h * 0.70), w, h))
            mix = Image.new("L", (w, top.height + bottom.height))
            mix.paste(top, (0, 0))
            mix.paste(bottom, (0, top.height))
            thumb = mix.resize((64, 32))
            return hashlib.sha1(thumb.tobytes()).hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _next_pending_task(task_plan: List[Dict[str, Any]], completed_tasks: List[int]) -> Optional[Dict[str, Any]]:
        done = set(int(x) for x in completed_tasks)
        for item in task_plan:
            tid = int(item.get("id", -1))
            if tid not in done:
                return item
        return task_plan[-1] if task_plan else None

    def _plan_to_text(self) -> str:
        if not self._task_plan:
            return "无"
        lines = []
        done = set(self.state.completed_tasks)
        for item in self._task_plan:
            step_id = item.get("id", "?")
            stage = item.get("stage", "")
            goal = item.get("goal", "")
            action_hint = item.get("action_hint", "")
            mark = "[DONE]" if step_id in done else "[TODO]"
            lines.append(f"{mark} {step_id}. {stage} | {goal} | hint={action_hint}")
        return "\\n".join(lines)

    def _build_prompt(
                self,
                instruction: str,
                history_actions: list,
                reviewer_feedback: str = "",
                retry_count: int = 0,
                app_name: str = "",
                completed_tasks: Optional[List[int]] = None,
                active_task: Optional[Dict[str, Any]] = None,
        ) -> str:
            self._current_instruction = instruction or ""
            recent = self._recent_history(history_actions, window=2)
            completed_tasks = completed_tasks or []
            pending = [x for x in self._task_plan if int(x.get("id", -1)) not in set(completed_tasks)]
            active_task_text = "无"
            if active_task:
                active_task_text = f"#{active_task.get('id')} {active_task.get('stage')} | {active_task.get('goal')} | hint={active_task.get('action_hint')}"

            review_section = ""
            if reviewer_feedback and reviewer_feedback != "PASS":
                review_section = f"""
    [Reviewer 反馈]
    - 当前重试次数: {retry_count}
    - 审核意见: {reviewer_feedback}
    - 请绝对服从审核意见，优先修复违规行为。
    """
            return f"""你是 Android GUI 智能代理。每步按 ReAct：观察 -> 分析 -> 动作。
    任务目标：【{instruction}】

    [执行上下文]
    - 目标 App: {app_name or self._app_name or '未知'}
    - 历史摘要: {self.state.get_summary()}
    - 最近动作: {recent}
    - 步数: {self.state.step_count}/{self.state.max_steps}
    - 总任务列表:
    {self._plan_to_text()}
    - 已完成任务ID: {completed_tasks}
    - 未完成任务数: {len(pending)}
    - 当前优先子任务: {active_task_text}
    {review_section}

    [视觉认知常识]
    A) 斜放的放大镜（🔎/🔍）代表【搜索入口】；漏斗图标代表【筛选入口】。
    B) 如果任务涉及具体日期（如明天、后天），请基于界面上标注有“今天”字样的日期进行相对推算。

    [业务规则与决策建议]
    1) [输入与防诱导]：在执行 TYPE 前，必须先 CLICK 激活目标输入框（寻找到闪烁的竖线光标）。
       -> 🚨 绝对禁止：如果下方出现了包含目标词的“搜索历史”或“推荐词”，绝对不要去点击它！必须强行对自己激活的输入框执行 TYPE 动作！
    2) [地名脱敏]：如果是打车、导航等出行任务，TYPE 输入地名时，切勿带有城市前缀/后缀（例如指令是“西安回民街”，你只能 TYPE "回民街"）。
    3) [动作闭环]：TYPE 完成后，必须寻找并精准 CLICK 页面原生 UI 中的“提交”、“确认”、“搜索”按钮完成闭环。
    4) [外卖与购物]：当搜索到目标商品或外卖时，不要去点击商品图片或标题，必须去 CLICK 旁边的“选规格”、“加入购物车”或“+”号按钮。
    5) [播放优先]：若当前优先子任务是播放/查看类任务，且界面中出现了目标对应的特定“播放按钮/三角图标”，优先 CLICK 该播放按钮，而不是点整体卡片。
    6) [隐私与完结检测]：一旦画面进入【支付界面】、【结账页面】或打车任务的【确认呼叫界面】，由于涉及隐私风险，不要进行真实支付或呼叫，请立刻输出 COMPLETE:[]。
    7) [美团导航专项]：若任务是点外卖且当前在‘美团’APP 首页，第一步必须是寻找并点击‘外卖’图标入口。
    8) 若当前还未进入目标应用，优先使用 OPEN 打开目标App。

    [合法动作]
    CLICK / TYPE / SCROLL / OPEN / COMPLETE

    [输出格式]
    必须严格按以下结构输出（不可遗漏）：
    [State] 当前阶段
    [Observe] 只描述当前屏幕可见事实
    [Analyze] 结合当前状态，判断下一步意图
    [Action] 仅一条动作，格式如下之一：
    CLICK:[[x,y]] 或 CLICK:[id]
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

    def _normalize_output(self, action: str, params: dict, som_map: Optional[Dict[int, Dict[str, Any]]] = None) -> \
    Tuple[str, dict, str]:
        params = params or {}
        som_map = som_map or {}

        if action == "CLICK_ID":
            idx = params.get("id") if isinstance(params, dict) else None
            try:
                idx = int(idx)
            except Exception:
                idx = None
            if idx and idx in som_map:
                center = som_map[idx].get("center", [500, 500])
                return "CLICK", {"point": center}, "根据编号点击目标控件"
            return "CLICK", {"point": [500, 500]}, "编号未命中，兜底点击"

        if action not in self.VALID_ACTIONS:
            return "CLICK", {"point": [500, 500]}, "兜底点击"

        return action, params, ""

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Limit reached")

        current_signature = self._image_signature(input_data.current_image)
        current_focus_signature = self._focus_signature(input_data.current_image)

        initial_state: WorkflowState = {
            "input_data": input_data,
            "plan_instruction": self._plan_instruction,
            "app_name": self.state.app_name or self._app_name,
            "task_plan": self._task_plan,
            "history_actions": input_data.history_actions,
            "completed_tasks": list(self.state.completed_tasks),
            "last_completed_update_step": self.state.last_completed_update_step,
            "frame_hash_global": self.state.last_visual_hash,
            "frame_hash_local": self.state.last_visual_hash_local,
            "hash_changed": True,
            "proposed_action": "",
            "proposed_params": {},
            "model_effect": "",
            "reviewer_decision": "",
            "reviewer_feedback": "",
            "violation_code": "",
            "candidate_completed_task_ids": [],
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
            logger.warning(f"Graph invoke failed: {e}")
            return AgentOutput(action="CLICK", parameters={"point": [500, 500]}, raw_output=f"Error: {e}")

        self.state.app_name = final_state.get("app_name", self.state.app_name or self._app_name)
        self.state.task_plan = list(final_state.get("task_plan") or self._task_plan)
        self.state.completed_tasks = list(final_state.get("completed_tasks") or self.state.completed_tasks)
        self.state.last_completed_update_step = int(
            final_state.get("last_completed_update_step", self.state.last_completed_update_step))

        self.state.update(
            f"{output.action}:{output.parameters}",
            expected_effect or "画面发生变化",
            visual_hash=final_state.get("frame_hash_global", current_signature),
            visual_hash_local=final_state.get("frame_hash_local", current_focus_signature),
            visual_changed=bool(final_state.get("hash_changed", True)),
        )
        self.state.last_image = input_data.current_image.copy()

        return output