# 文件路径: agent.py
import os
import re
import logging
from typing import Tuple, Dict, Any, List
from agent_base import BaseAgent, AgentOutput, AgentInput
from utils.vision_enhancer import add_coordinate_grid

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    中兴捧月 GUI Agent - V2 增强版 (视觉网格 + 防死循环状态机 + Reflexion)
    """

    def _initialize(self):
        """初始化内部状态机"""
        self.step_history_text = []

        # --- V2 新增：防死循环监控变量 ---
        self.last_action_type = None
        self.last_action_params = {}
        self.consecutive_stuck_count = 0

    def reset(self):
        """每个测试用例开始前重置状态，满足评测器规范"""
        self.step_history_text.clear()
        self.last_action_type = None
        self.last_action_params = {}
        self.consecutive_stuck_count = 0

    def _detect_loop(self, current_action: str, current_params: dict):
        """
        V2 新增：检测是否陷入了重复点击等死循环
        在 1000x1000 坐标系下，如果连续点击的距离极近，视为卡死。
        """
        if current_action == self.last_action_type == "CLICK":
            p_curr = current_params.get("point", [0, 0])
            p_last = self.last_action_params.get("point", [0, 0])

            # 计算曼哈顿距离，如果在 50 (5%屏幕) 范围内，视为重复无效点击
            dist = abs(p_curr[0] - p_last[0]) + abs(p_curr[1] - p_last[1])
            if dist < 50:
                self.consecutive_stuck_count += 1
            else:
                self.consecutive_stuck_count = 0
        else:
            self.consecutive_stuck_count = 0

        self.last_action_type = current_action
        self.last_action_params = current_params

    def _build_advanced_prompt(self, instruction: str, history: List[str]) -> str:
        """
        构建带有动态警告和严格 Few-shot 的系统提示词
        """
        history_str = "\n".join(history) if history else "No previous actions. This is step 1."

        # --- V2 新增：动态警告注入 ---
        warning_block = ""
        if self.consecutive_stuck_count >= 2:
            warning_block = """
🚨 [CRITICAL WARNING]: 
You are repeating the SAME CLICK ACTION and making no progress! The UI element might be unclickable or you are stuck. 
YOU MUST CHANGE YOUR STRATEGY NOW. Try to SCROLL the screen to find new elements, or click a completely different area!
"""

        return f"""You are an expert GUI Agent for mobile devices. 
Your ultimate task is: 【{instruction}】

[Image Information]
The screen has a 10x10 red grid overlay. Each cell is 100x100 in the normalized 1000x1000 system. 
X-axis: 0 (left) to 1000 (right). Y-axis: 0 (top) to 1000 (bottom).

[Action History]
{history_str}
{warning_block}

[Reasoning Protocol (ReAct)]
You MUST formulate your step in the following strict order:
Thought:
- Observation: What changed after the last action? Is the target visible?
- Plan: What is the single next move?
- Coordinate Estimation: Use the grid to estimate X and Y.
Action: <Exactly one action format>

[Action Formats & STRICT Rules]
1. CLICK:[[x, y]]  (e.g., CLICK:[[500, 150]])
2. TYPE:['content'] 
3. OPEN:['app name']
4. COMPLETE:[]
5. SCROLL:[[start_x, start_y], [end_x, end_y]]
   -> RULE FOR SCROLLING: To scroll the page DOWN (see lower content), you must swipe UP. Example: SCROLL:[[500, 800], [500, 200]].

Now, process the current screen and output your Thought and Action.
"""

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        # 给原图加上坐标网格 (依赖我们上一版的 utils/vision_enhancer.py)
        enhanced_image = add_coordinate_grid(input_data.current_image)
        final_prompt = self._build_advanced_prompt(input_data.instruction, self.step_history_text)

        return [
            {"role": "user", "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(enhanced_image)}}
            ]}
        ]

    def act(self, input_data: AgentInput) -> AgentOutput:
        messages = self.generate_messages(input_data)

        try:
            # 降低温度，追求极致的逻辑确定性
            response = self._call_api(messages, temperature=0.0)
            raw_output = response.choices[0].message.content
            usage_info = self.extract_usage_info(response)

            action, params = self._robust_parse(raw_output)

            # --- V2 新增：记录文本历史并进行死循环检测 ---
            log_str = f"Step {input_data.step_count}: {action} {params}"
            self.step_history_text.append(log_str)
            self._detect_loop(action, params)

            return AgentOutput(
                action=action,
                parameters=params,
                raw_output=raw_output,
                usage=usage_info
            )

        except Exception as e:
            logger.error(f"[Agent Error]: {e}")
            # 异常兜底策略
            return AgentOutput(action="CLICK", parameters={"point": [500, 500]}, raw_output=str(e))

    def _robust_parse(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """保留上版本的安全解析器，它完全符合赛题要求的5种输出格式"""
        action_text = raw_text.split("Action:")[-1].strip() if "Action:" in raw_text else raw_text

        if 'COMPLETE' in action_text: return "COMPLETE", {}

        match_click = re.search(r'CLICK:\s*\[?\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_click: return "CLICK", {"point": [int(match_click.group(1)), int(match_click.group(2))]}

        match_scroll = re.search(r'SCROLL:\s*\[?\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_scroll: return "SCROLL", {
            "start_point": [int(match_scroll.group(1)), int(match_scroll.group(2))],
            "end_point": [int(match_scroll.group(3)), int(match_scroll.group(4))]
        }

        match_type = re.search(r'TYPE:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_type: return "TYPE", {"text": match_type.group(1)}

        match_open = re.search(r'OPEN:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_open: return "OPEN", {"app_name": match_open.group(1)}

        return "CLICK", {"point": [500, 500]}