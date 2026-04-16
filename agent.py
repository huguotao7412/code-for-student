# 文件路径: agent.py
from dotenv import load_dotenv

load_dotenv()

import hashlib
import json
import logging
import re
from typing import Any, Dict, Tuple, cast

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - fallback for environments without langgraph
    END = None
    START = None
    StateGraph = None

from agent_base import AgentInput, AgentOutput, BaseAgent, ConfigTamperError, FORBIDDEN_KWARGS
from utils.a2a_protocol import A2AChannels, read_payload
from utils.graph_state import WorkflowState
from utils.nodes import actor_node, format_output_node, planner_node, reviewer_node
from utils.parser import robust_parse
from utils.state import GUIState

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """LLM 主导，最小 schema 适配。"""

    VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE"}
    REVIEW_MAX_RETRY = 1
    TASK_TYPE_KEYWORDS = {
        "map": ["地图", "导航", "打车", "路线", "起点", "终点"],
        "flight": ["航班", "机票", "出发", "到达", "去哪儿旅行", "去哪旅行"],
        "food": ["外卖", "美团", "下单", "购买", "店铺"],
        "video": ["视频", "播放", "第", "集", "腾讯视频", "爱奇艺", "哔哩", "bilibili"],
        "search": ["搜索", "查找", "检索", "搜"],
    }
    MAP_CITY_PREFIXES = (
        "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "武汉",
        "天津", "西安", "长沙", "郑州", "青岛", "厦门", "宁波", "合肥", "福州", "济南",
        "昆明", "沈阳", "大连", "南昌", "贵阳", "南宁", "石家庄", "太原", "哈尔滨", "长春",
    )
    APP_ALIASES = {
        "美团外卖": "美团",
        "去哪旅行": "去哪儿旅行",
    }
    PROMPT_CORE_RULES = [
        "- 屏幕截图上已用红框/蓝框标记了带数字 ID 的控件。强烈优先使用 CLICK_ID:[数字] 进行精确点击！",
        "- 只有在目标位置确实没有数字标记框时，才退化使用 CLICK:[[x,y]] (0-1000归一化坐标)。",
        "- 计划赶不上变化：当前的“任务计划”仅供参考。如果遇到弹窗、广告、权限请求遮挡，必须立刻中断原计划，本步的唯一目标是关闭遮挡物！",
        "- 所有判断以当前截图可见控件为准，绝对不依赖固定坐标或凭空想象的页面模板。",
        "- TYPE 文本时，必须剥离任务指令中的动词前缀（如“去”、“导航到”、“搜索”），仅输入核心实体名词。若需精确模糊匹配，可自行使用 .* 作为前缀。",
        "- 若刚执行 TYPE，下一步优先 ENTER 或点击搜索/确认控件。",
        "- 只在目标状态明确达成时才输出 COMPLETE。",
    ]

    PROMPT_STRATEGIES = {
        "search": [
            "- 搜索链路必须是：激活搜索框 -> TYPE 任务词 -> 搜索确认。",
            "- 内容区出现同名词时，也不能跳过 TYPE 直接点内容。",
            "- 搜索入口可为文字按钮、放大镜图标或斜放的搜索图标。",
        ],
        "map": [
            "- 地图双输入链路：起点确认后，先进入终点输入入口，再 TYPE 终点。",
            "- 终点入口提示文案不等于可输入框；未激活输入框前不得直接 TYPE。",
            "- 地点词可在证据充分时去城市前缀；如回民街可用正则形式。",
        ],
        "video": [
            "- 播放第N集任务：先进入播放态，再选集。",
            "- 评论区/讨论区任务：先进入播放态，再进入评论/讨论区域。",
        ],
        "publish": [
            "- 发布类任务：先激活输入框，再 TYPE，再点击发送/发布。",
        ],
        "date": [
            "- 日期任务先识别界面上今天对应日期，再顺推明天/后天。",
        ],
        "voice": [
            "- 更换语音包任务先确认目标词条可见，再点击对应词条。",
        ],
        "dedupe": [
            "- 若任务是去喜欢里搜索且当前已在喜欢页，不要重复点击喜欢入口。",
        ],
        "app_alias": [
            "- 任务提到美团外卖时，OPEN 应用名使用 美团。",
            "- 任务提到去哪旅行时，OPEN 应用名使用 去哪儿旅行。",
        ],
    }

    def _initialize(self):
        self.state = GUIState(max_steps=45)
        self._plan_instruction = ""
        self._task_plan = []
        self._current_instruction = ""
        self._openai_client = None
        self.graph = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            logger.warning("langgraph is not available, fallback to internal multi-agent runtime")
            return None
        workflow = StateGraph(cast(Any, WorkflowState))

        workflow.add_node("planner", lambda state: planner_node(state, self))
        workflow.add_node("actor", lambda state: actor_node(state, self))
        workflow.add_node("reviewer", lambda state: reviewer_node(state, self))
        workflow.add_node("format_output", lambda state: format_output_node(state, self))

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "actor")

        def _actor_route(state: WorkflowState) -> str:
            action = str(state.get("proposed_action", "")).upper()
            params = state.get("proposed_params", {}) or {}
            retry_count = int(state.get("retry_count", 0))
            if self._should_fast_pass_review(action, params, retry_count):
                return "format_output"
            return "reviewer"

        workflow.add_conditional_edges(
            "actor",
            _actor_route,
            {
                "reviewer": "reviewer",
                "format_output": "format_output",
            },
        )

        def _review_route(state: WorkflowState) -> str:
            review_payload = read_payload(state, A2AChannels.REVIEW, default={}) or {}
            verdict = str(review_payload.get("verdict", "")).upper()
            retry_count = int(review_payload.get("retry_count", state.get("retry_count", 0)))
            if verdict == "PASS" or state.get("reviewer_feedback") == "PASS" or retry_count > self.REVIEW_MAX_RETRY:
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

    def _build_plan_prompt(self, instruction: str) -> str:
        return f"""你是 GUI 任务规划器。请先理解任务，再输出 4~8 条结构化子步骤。
要求：
1) 仅输出 JSON，不要解释。
2) JSON 格式：
{{
  \"sub_steps\": [
    {{\"id\": 1, \"stage\": \"阶段名\", \"goal\": \"要完成什么\", \"action_hint\": \"OPEN/CLICK/TYPE/SCROLL/COMPLETE\", \"input_text\": \"如果是TYPE步骤，请填入提取出的纯净核心名词\"}}
  ]
}}
3) 计划要可迁移，不依赖固定坐标、固定 App 或固定页面模板。
4) 优先使用任务意图驱动（例如：搜索、输入确认、结果选择、发布、播放、地图起终点）。
5) TYPE 步骤必须聪明地提取：剥离“去”、“导航到”、“搜索”等行为动词，仅保留核心实体名词。
6) 搜索任务必须拆成“激活搜索框 -> TYPE任务词 -> 搜索确认”，不能用点击内容区同名词替代 TYPE。
7) 应用名归一化：任务提到“美团外卖”时先 OPEN “美团”再进入外卖入口；任务提到“去哪旅行”时使用“去哪儿旅行”。
8) 搜索入口可为文字按钮、标准放大镜或斜放的🔍/🔎图标。
9) 日期任务（今天/明天/后天）先识别界面已展示日期（如“今天”），再顺推对应日期。

通用示例：
[示例1：搜索并查看]
任务：在某内容应用中搜索关键词并打开一个结果
{{\"sub_steps\":[
{{\"id\":1,\"stage\":\"进入任务场景\",\"goal\":\"进入目标应用或目标页面\",\"action_hint\":\"OPEN\"}},
{{\"id\":2,\"stage\":\"处理遮挡\",\"goal\":\"关闭弹窗/权限引导\",\"action_hint\":\"CLICK\"}},
{{\"id\":3,\"stage\":\"进入搜索\",\"goal\":\"点击搜索入口并激活输入框\",\"action_hint\":\"CLICK\"}},
{{\"id\":4,\"stage\":\"输入关键词\",\"goal\":\"输入任务关键词\",\"action_hint\":\"TYPE\",\"input_text\":\"任务原文关键词\"}},
{{\"id\":5,\"stage\":\"确认搜索\",\"goal\":\"执行搜索确认\",\"action_hint\":\"CLICK\"}},
{{\"id\":6,\"stage\":\"选择结果\",\"goal\":\"进入目标内容\",\"action_hint\":\"CLICK\"}},
{{\"id\":7,\"stage\":\"结束\",\"goal\":\"任务完成\",\"action_hint\":\"COMPLETE\"}}
]}}

[示例2：地图起终点]
任务：在地图应用中输入起点和终点并进入路线/打车结果
{{\"sub_steps\":[
{{\"id\":1,\"stage\":\"进入功能入口\",\"goal\":\"进入路线或打车功能\",\"action_hint\":\"CLICK\"}},
{{\"id\":2,\"stage\":\"输入起点\",\"goal\":\"输入并选择起点候选项\",\"action_hint\":\"TYPE\",\"input_text\":\"任务原文起点\"}},
{{\"id\":3,\"stage\":\"确认起点\",\"goal\":\"点击候选项确认起点\",\"action_hint\":\"CLICK\"}},
{{\"id\":4,\"stage\":\"进入终点输入\",\"goal\":\"切换到终点输入入口\",\"action_hint\":\"CLICK\"}},
{{\"id\":5,\"stage\":\"输入终点\",\"goal\":\"输入并选择终点候选项\",\"action_hint\":\"TYPE\",\"input_text\":\"任务原文终点\"}},
{{\"id\":6,\"stage\":\"结束\",\"goal\":\"路线/打车结果达成\",\"action_hint\":\"COMPLETE\"}}
]}}

[示例3：发布型任务]
任务：在内容页面发布评论/消息
{{\"sub_steps\":[
{{\"id\":1,\"stage\":\"定位入口\",\"goal\":\"进入目标内容页并找到评论/输入入口\",\"action_hint\":\"CLICK\"}},
{{\"id\":2,\"stage\":\"激活输入\",\"goal\":\"点击输入框使其可输入\",\"action_hint\":\"CLICK\"}},
{{\"id\":3,\"stage\":\"输入内容\",\"goal\":\"输入评论或消息文本\",\"action_hint\":\"TYPE\",\"input_text\":\"任务原文内容\"}},
{{\"id\":4,\"stage\":\"提交\",\"goal\":\"点击发送/发布\",\"action_hint\":\"CLICK\"}},
{{\"id\":5,\"stage\":\"结束\",\"goal\":\"确认已发布\",\"action_hint\":\"COMPLETE\"}}
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
            input_text = item.get("input_text", "")
            extra = f" | input={input_text}" if input_text else ""
            lines.append(f"{step_id}. {stage} | {goal} | hint={action_hint}{extra}")
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
                    "input_text": str(item.get("input_text", "")).strip(),
                })
            if 4 <= len(parsed) <= 8:
                return parsed
        except Exception:
            pass

        # 解析失败时，退化为可用的通用计划，避免首步无计划。
        return [
            {"id": 1, "stage": "打开或进入目标应用", "goal": "到达任务主场景", "action_hint": "OPEN", "input_text": ""},
            {"id": 2, "stage": "定位入口", "goal": "找到搜索/功能入口", "action_hint": "CLICK", "input_text": ""},
            {"id": 3, "stage": "输入", "goal": "提取并原样输入任务关键词", "action_hint": "TYPE", "input_text": "任务原文关键词"},
            {"id": 4, "stage": "确认", "goal": "确认搜索或下一步", "action_hint": "CLICK", "input_text": ""},
            {"id": 5, "stage": "结果处理", "goal": "进入目标结果页面", "action_hint": "CLICK", "input_text": ""},
            {"id": 6, "stage": "完成", "goal": "任务结束", "action_hint": "COMPLETE", "input_text": ""},
        ]

    def _ensure_task_plan(self, instruction: str, current_image, current_image_url: str = "") -> None:
        if self._plan_instruction == instruction and self._task_plan:
            return

        self._plan_instruction = instruction
        self._task_plan = []
        plan_prompt = self._build_plan_prompt(instruction)
        image_url = current_image_url or self._encode_image(current_image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": plan_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]
        try:
            response = self._call_api(messages)
            raw_plan = response.choices[0].message.content
            self._task_plan = self._parse_plan(raw_plan)
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            self._task_plan = self._parse_plan("")

    def _app_specific_hints(self, instruction: str, task_type: str = "general", flow_flags: Dict[str, Any] | None = None) -> str:
        """将关键策略模块化，减少散落硬编码。"""
        text = instruction or ""
        flags = flow_flags or {}
        hints = list(self.PROMPT_CORE_RULES)

        if task_type in {"search", "food", "flight", "map"} or any(k in text for k in ["搜索", "查找", "检索", "搜"]):
            hints.extend(self.PROMPT_STRATEGIES["search"])
        if task_type == "map" or any(k in text for k in ["地图", "导航", "打车", "路线", "起点", "终点"]):
            hints.extend(self.PROMPT_STRATEGIES["map"])
        if task_type == "video" or any(k in text for k in ["播放", "收听", "听", "观看"]):
            hints.extend(self.PROMPT_STRATEGIES["video"])
        if any(k in text for k in ["评论", "发布", "发送", "提交"]):
            hints.extend(self.PROMPT_STRATEGIES["publish"])
        if any(k in text for k in ["今天", "明天", "后天", "日期", "出发", "入住"]):
            hints.extend(self.PROMPT_STRATEGIES["date"])
        if any(k in text for k in ["语音包", "导航语音", "语音", "更换语音"]):
            hints.extend(self.PROMPT_STRATEGIES["voice"])
        if "喜欢" in text:
            hints.extend(self.PROMPT_STRATEGIES["dedupe"])
        if "美团外卖" in text or "去哪旅行" in text:
            hints.extend(self.PROMPT_STRATEGIES["app_alias"])
        if flags.get("search_reactivate_needed"):
            hints.append("- 你很可能已进入新搜索页：先重新激活搜索框，再 TYPE。")

        return "\n".join(hints)

    def _build_prompt(
                self,
                instruction: str,
                history_actions: list,
                reviewer_feedback: str = "",
                retry_count: int = 0,
        ) -> str:
            stuck_warning = ""
            if self.state.stuck_level >= 2:
                stuck_warning = "\n[系统警告] 你似乎陷入了死循环，画面可能没有变化！请立即改变策略（如：换一个入口、尝试关闭潜在的透明弹窗、或执行一次 SCROLL 滑动以刷新页面）。"

            self._current_instruction = instruction or ""
            recent = self._recent_history(history_actions, window=2)
            task_type, task_slots, flow_flags = self._build_task_profile(instruction, history_actions)

            # === 动态反思链增强：提取上一步的期望效果 ===
            last_expected_effect = "无（当前为第一步）"
            if history_actions and isinstance(history_actions[-1], dict):
                last_raw = history_actions[-1].get("raw_output", "")
                if "[Expected Effect]" in last_raw:
                    last_expected_effect = last_raw.split("[Expected Effect]")[-1].strip()
            # ==========================================

            review_section = ""
            if reviewer_feedback and reviewer_feedback != "PASS":
                review_section = f"""
    [Reviewer 反馈]
    - 当前重试次数: {retry_count}
    - 审核意见: {reviewer_feedback}
    - 必须先修复该问题，再给下一步动作。
    """

            policy = self._app_specific_hints(instruction, task_type=task_type, flow_flags=flow_flags)

            return f"""你是 Android GUI 智能代理。每一步必须按 ReAct 框架进行严谨的逻辑推理。
    任务目标：【{instruction}】

    [当前状态]
    - 历史摘要: {self.state.get_summary()}
    - 最近动作: {recent}
    - 上一步期望看到的变化: {last_expected_effect}
    - 步数: {self.state.step_count}/{self.state.max_steps}
    - 当前宏观任务计划(仅供参考，遇遮挡须灵活变通):
    {self._plan_to_text()}
    {review_section}
    {stuck_warning}

    [合法动作]
    CLICK_ID / CLICK / TYPE / SCROLL / OPEN / ENTER / COMPLETE

    [关键策略]
    {policy}

    [任务画像]
    - task_type={task_type}
    - task_slots={task_slots}
    - flow_flags={flow_flags}

    [坐标系统]
    当前屏幕坐标为归一化 1000x1000：左上 [0,0]，右下 [1000,1000]。
    如果目标没有数字ID，CLICK 和 SCROLL 的坐标必须在 0~1000。

    [输出格式]
    [Observe] 1. 当前画面最核心的特征是什么？ 2. 仔细检查：画面中是否有浮层弹窗、广告、或者权限申请？
    [Analyze] 1. 反思：当前画面是否达成了“上一步期望看到的变化”？ 2. 决策：如果有弹窗遮挡，本步最高优先级是找到并点击它的关闭/跳过按钮；如果没有遮挡，结合宏观计划，决定现在的具体目标。
    [Action] 仅一条动作，格式如下之一：
    CLICK_ID:[数字ID]
    CLICK:[[x,y]]
    TYPE:['文本']
    SCROLL:[[x1,y1],[x2,y2]]
    OPEN:['应用名']
    ENTER:[]
    COMPLETE:[]
    [Expected Effect] 预判该动作执行后，画面应发生什么变化
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
        return [max(0, min(1000, int(point[0]))), max(0, min(1000, int(point[1])))]

    @staticmethod
    def _normalize_text(text: Any) -> str:
        raw = "" if text is None else str(text)
        # 只去掉模型常见外层包装，不碰正文。
        return raw.strip().strip("'\" ")

    def _detect_task_type(self, instruction: str) -> str:
        text = instruction or ""
        if any(k in text for k in self.TASK_TYPE_KEYWORDS["map"]):
            return "map"
        if any(k in text for k in self.TASK_TYPE_KEYWORDS["flight"]):
            return "flight"
        if any(k in text for k in self.TASK_TYPE_KEYWORDS["food"]):
            return "food"
        if any(k in text for k in self.TASK_TYPE_KEYWORDS["video"]):
            return "video"
        if any(k in text for k in self.TASK_TYPE_KEYWORDS["search"]):
            return "search"
        return "general"

    def _extract_map_slots(self, instruction: str) -> Dict[str, str]:
        text = instruction or ""
        slots: Dict[str, str] = {}

        direct = re.search(r"从(.+?)去(.+?)(?:，|。|,|$)", text)
        if direct:
            slots["from"] = direct.group(1).strip()
            slots["to"] = direct.group(2).strip()
        else:
            dest = re.search(r"去(.+?)(?:，|。|,|$)", text)
            if dest:
                slots["to"] = dest.group(1).strip()

        for key in ("from", "to"):
            value = slots.get(key, "")
            for city in self.MAP_CITY_PREFIXES:
                if value.startswith(city) and len(value) > len(city) + 1:
                    slots[f"{key}_core"] = value[len(city):].strip()
                    break
        return slots

    def _build_task_profile(self, instruction: str, history_actions: list) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        task_type = self._detect_task_type(instruction)
        slots: Dict[str, str] = {}
        if task_type == "map":
            slots.update(self._extract_map_slots(instruction))

        hist = history_actions or []

        def _hist_click_point(item: Dict[str, Any]) -> tuple[int, int] | None:
            if not isinstance(item, dict) or str(item.get("action", "")).upper() != "CLICK":
                return None
            params = item.get("parameters", {}) if isinstance(item.get("parameters", {}), dict) else {}
            point = params.get("point")
            if isinstance(point, list) and len(point) == 2:
                try:
                    return int(point[0]), int(point[1])
                except Exception:
                    return None
            return None

        typed_count = sum(
            1
            for item in hist
            if isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        )
        click_count = sum(
            1
            for item in hist
            if isinstance(item, dict) and str(item.get("action", "")).upper() == "CLICK"
        )

        has_episode_target = bool(re.search(r"第\s*[一二三四五六七八九十\d]+\s*集", instruction or ""))
        needs_comment_area = any(k in (instruction or "") for k in ["评论区", "讨论区", "评论", "讨论"])
        is_voice_package_task = any(k in (instruction or "") for k in ["语音包", "导航语音", "语音", "更换语音"])

        search_like = task_type in {"search", "food", "flight", "map"} or any(
            k in (instruction or "") for k in ["搜索", "查找", "检索", "搜"]
        )
        search_entry_clicked = search_like and click_count >= 1
        search_reactivate_needed = bool(search_entry_clicked and typed_count == 0)

        flow_flags = {
            "has_typed": typed_count > 0,
            "typed_count": typed_count,
            "click_count": click_count,
            "has_episode_target": has_episode_target,
            "needs_comment_area": needs_comment_area,
            "is_voice_package_task": is_voice_package_task,
            "search_reactivate_needed": search_reactivate_needed,
        }
        return task_type, slots, flow_flags

    def _self_check_type_text(self, text: str) -> str:
        """轻量自检：移除硬编码地点词，信任模型提取的核心名词，仅做基础补全和兜底。"""
        text = self._normalize_text(text)
        instruction = getattr(self, "_current_instruction", "") or ""
        if not text or not instruction:
            return text

        # 泛化策略：若指令里存在书名号短语（如《狂飙》），且当前文本是其子串，优先补全为完整短语
        for match in re.findall(r"《[^》]+》[^，。！？；,!?;\n]*", instruction):
            phrase = match.strip()
            if text and text in phrase and len(text) < len(phrase):
                return phrase

        # 如果大模型加上了 .* 前缀且主体在指令中，信任大模型的正则决策
        if text.startswith(".*") and text[2:] in instruction:
            return text

        # 兜底：如果模型提取的文本确实在指令里，原样返回即可
        if text in instruction:
            return text

        # 极其轻量的安全剥离：去除最常见的错误前缀，不要写死城市名
        for prefix in ["去", "到", "前往", "导航到", "打车到", "搜索", "查找"]:
            if text.startswith(prefix) and len(text) > len(prefix):
                candidate = text[len(prefix):].strip()
                if candidate and candidate in instruction:
                    return candidate

        return text

    def _normalize_output(self, action: str, params: dict) -> Tuple[str, dict, str]:
        params = params or {}

        if action == "CLICK_ID":
            # 赛题标准输出要求 CLICK:[[x,y]]，我们在这里拦截 CLICK_ID 并转换为实际归一化坐标
            element_id = params.get("id", params.get("selected_id", -1))
            try:
                element_id = int(element_id)
            except (ValueError, TypeError):
                element_id = -1

            if hasattr(self, "current_element_map") and element_id in self.current_element_map:
                meta = self.current_element_map[element_id]
                center = None

                # 兼容不同的 element_map 数据结构 (字典结构含 center/bbox，或纯列表)
                if isinstance(meta, dict):
                    center = meta.get("center") or meta.get("point")
                    if not center and "bbox" in meta:
                        x0, y0, x1, y1 = meta["bbox"]
                        center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]
                elif isinstance(meta, (list, tuple)):
                    if len(meta) == 2:
                        center = list(meta)
                    elif len(meta) == 4:
                        x0, y0, x1, y1 = meta
                        center = [int((x0 + x1) / 2), int((y0 + y1) / 2)]

                if center:
                    clipped_point = self._clip_norm_point(center)
                    return "CLICK", {"point": clipped_point}, f"精确映射控件 ID:{element_id} -> 坐标 {clipped_point}"

            # 如果没找到对应 ID 或者解析失败，退化为屏幕中央点击
            logger.warning(f"CLICK_ID {element_id} 转换坐标失败，退化为中心点击")
            return "CLICK", {"point": [500, 500]}, f"ID {element_id} 未找到，兜底点击"

        if action == "ENTER":
            if isinstance(params, dict) and "point" in params:
                return "CLICK", {"point": self._clip_norm_point(params["point"])}, "确认输入并继续"
            if isinstance(params, dict) and "x" in params and "y" in params:
                return "CLICK", {"point": self._clip_norm_point([params["x"], params["y"]])}, "确认输入并继续"
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
            normalized_name = self._normalize_text(app_name)
            normalized_name = self.APP_ALIASES.get(normalized_name, normalized_name)
            if not normalized_name:
                instruction = getattr(self, "_current_instruction", "") or ""
                app_candidates = [
                    "美团", "美团外卖", "去哪儿旅行", "去哪旅行", "百度地图", "高德地图", "腾讯视频", "爱奇艺",
                    "哔哩哔哩", "Bilibili", "抖音", "快手", "芒果TV", "喜马拉雅", "QQ音乐", "微博",
                ]
                for app in app_candidates:
                    if app in instruction:
                        normalized_name = self.APP_ALIASES.get(app, app)
                        break
            if normalized_name:
                return "OPEN", {"app_name": normalized_name}, ""
            return "CLICK", {"point": [500, 500]}, "OPEN 缺少 app_name，退化为安全点击"

        if action == "COMPLETE":
            return "COMPLETE", {}, ""

        if action not in self.VALID_ACTIONS:
            return "CLICK", {"point": [500, 500]}, "兜底点击"

        return action, params, ""


    def _build_initial_state(self, input_data: AgentInput) -> WorkflowState:
        task_type, task_slots, flow_flags = self._build_task_profile(input_data.instruction, input_data.history_actions)

        # === 新增：调用 SoM 视觉增强 ===
        self.current_element_map = {}
        encoded_image_url = self._encode_image(input_data.current_image)
        som_image_url = encoded_image_url
        try:
            from utils.ui_detector import draw_som_labels

            # 对当前图像进行画框标记，返回 画了框的图片 和 控件字典
            annotated_img, element_map = draw_som_labels(input_data.current_image)
            som_image_url = self._encode_image(annotated_img)
            self.current_element_map = element_map  # 暂存在实例上，供后续 CLICK_ID 转换为坐标使用
        except Exception as e:
            logger.warning(f"SoM detection failed: {e}")
            som_image_url = encoded_image_url
            self.current_element_map = {}
        # ================================

        state: Dict[str, Any] = {
            "input_data": input_data,
            "plan_instruction": self._plan_instruction,
            "task_plan": self._task_plan,
            "history_actions": input_data.history_actions,
            "mailbox": {},
            "task_type": task_type,
            "task_slots": task_slots,
            "flow_flags": flow_flags,
            "encoded_image_url": encoded_image_url,
            "current_image_url": som_image_url,  # <-- 传入带有数字框标记的图片给大模型
            "proposed_action": "",
            "proposed_params": {},
            "model_effect": "",
            "reviewer_feedback": "",
            "retry_count": 0,
            "review_source": "",
            "raw_output": "",
            "usage": None,
        }
        return cast(WorkflowState, cast(object, state))

    def _run_fallback_workflow(self, initial_state: WorkflowState) -> WorkflowState:
        # Graph 不可用时，按同一节点顺序执行，保持与多 agent 编排一致。
        state: Dict[str, Any] = dict(initial_state)
        workflow_state = cast(WorkflowState, cast(object, state))
        state.update(planner_node(workflow_state, self))
        state.update(actor_node(workflow_state, self))

        action = str(state.get("proposed_action", "")).upper()
        params = state.get("proposed_params", {}) or {}
        retry_count = int(state.get("retry_count", 0))
        if self._should_fast_pass_review(action, params, retry_count):
            state["reviewer_feedback"] = "PASS"
            state["review_source"] = "RULE_FAST_PASS"
        else:
            while True:
                state.update(reviewer_node(workflow_state, self))
                review_payload = read_payload(state, A2AChannels.REVIEW, default={}) or {}
                verdict = str(review_payload.get("verdict", "")).upper()
                retry_count = int(review_payload.get("retry_count", state.get("retry_count", 0)))
                if verdict == "PASS" or state.get("reviewer_feedback") == "PASS" or retry_count > self.REVIEW_MAX_RETRY:
                    break
                state.update(actor_node(workflow_state, self))

        state.update(format_output_node(workflow_state, self))
        return cast(WorkflowState, cast(object, state))

    def _execute_workflow(self, initial_state: WorkflowState) -> WorkflowState:
        if self.graph:
            try:
                return cast(WorkflowState, self.graph.invoke(cast(Any, initial_state)))
            except Exception as e:
                logger.warning(f"Graph invoke failed, fallback to internal workflow: {e}")
        return self._run_fallback_workflow(initial_state)

    def _finalize_output(self, final_state: WorkflowState) -> tuple[AgentOutput, str]:
        output = final_state.get("final_output")
        expected_effect = final_state.get("expected_effect", "画面发生变化")

        if not isinstance(output, AgentOutput):
            final_payload = read_payload(final_state, A2AChannels.FINAL_OUTPUT, default={}) or {}
            output = final_payload.get("final_output")
            expected_effect = final_payload.get("expected_effect", expected_effect)

        if not isinstance(output, AgentOutput):
            output = AgentOutput(
                action=final_state.get("normalized_action", "CLICK"),
                parameters=final_state.get("normalized_params", {"point": [500, 500]}),
                raw_output=final_state.get("raw_output", ""),
                usage=final_state.get("usage"),
            )
        return output, expected_effect

    def act(self, input_data: AgentInput) -> AgentOutput:
        if self.state.should_force_stop():
            return AgentOutput(action="COMPLETE", parameters={}, raw_output="Limit reached")

        current_signature = self._image_signature(input_data.current_image)
        initial_state = self._build_initial_state(input_data)

        try:
            final_state = self._execute_workflow(initial_state)
            output, expected_effect = self._finalize_output(final_state)
        except Exception as e:
            logger.warning(f"Workflow failed, use safe fallback action: {e}")
            output = AgentOutput(action="CLICK", parameters={"point": [500, 500]}, raw_output=f"Fallback: {e}")
            expected_effect = "兜底点击"

        self.state.update(
            f"{output.action}:{output.parameters}",
            expected_effect or "画面发生变化",
            visual_hash=current_signature,
        )
        self.state.last_image = input_data.current_image.copy()

        return output

    @staticmethod
    def _is_valid_norm_point(point: Any) -> bool:
        if not isinstance(point, list) or len(point) != 2:
            return False
        try:
            x = int(point[0])
            y = int(point[1])
        except Exception:
            return False
        return 0 <= x <= 1000 and 0 <= y <= 1000

    def _should_fast_pass_review(self, action: str, params: Dict[str, Any], retry_count: int) -> bool:
        if retry_count > 0:
            return False
        action = str(action or "").upper()
        if action == "CLICK":
            return self._is_valid_norm_point((params or {}).get("point"))
        if action == "SCROLL":
            p = params or {}
            return self._is_valid_norm_point(p.get("start_point")) and self._is_valid_norm_point(p.get("end_point"))
        return False

    def _get_step_image_url(self, state: Dict[str, Any], input_data: AgentInput) -> str:
        current_image_url = str(state.get("current_image_url", "") or "")
        if current_image_url:
            return current_image_url
        encoded_image_url = str(state.get("encoded_image_url", "") or "")
        if encoded_image_url:
            return encoded_image_url
        # 兜底仅编码一次，并回填到 state 供本步多节点复用。
        encoded_image_url = self._encode_image(input_data.current_image)
        state["encoded_image_url"] = encoded_image_url
        state["current_image_url"] = encoded_image_url
        return encoded_image_url

    def _get_openai_client(self):
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装 openai 包: pip install openai")
        self._openai_client = OpenAI(base_url=self._api_url, api_key=self._api_key)
        return self._openai_client

    def _call_api(self, messages, **kwargs):
        """Agent-local API wrapper: keep base safety checks but honor runtime kwargs."""
        forbidden_found = []
        safe_kwargs = {}
        for key, value in kwargs.items():
            if key.lower() in FORBIDDEN_KWARGS or key in FORBIDDEN_KWARGS:
                forbidden_found.append(key)
            else:
                safe_kwargs[key] = value

        if forbidden_found:
            logger.warning(
                f"[安全警告] 以下敏感参数已被移除: {forbidden_found}。"
                "请勿尝试传入 base_url、api_key、model 等参数。"
            )

        current_signature = self._compute_runtime_signature()
        if current_signature != self._config_signature:
            raise ConfigTamperError(
                "检测到配置篡改！运行时签名与初始化签名不一致。\n"
                f"初始签名: {self._config_signature}\n"
                f"当前签名: {current_signature}\n"
                "评测已终止。"
            )

        client = cast(Any, self._get_openai_client())

        logger.info(f"[API调用] model={self._model_id}, url={self._api_url}")

        user_extra_body = safe_kwargs.pop("extra_body", None)
        extra_body = {"thinking": {"type": "disabled"}}
        if isinstance(user_extra_body, dict):
            extra_body.update(user_extra_body)

        return client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            extra_body=extra_body,
            **safe_kwargs,
        )
