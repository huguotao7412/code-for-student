import os
from dotenv import load_dotenv
load_dotenv()

import re
import logging
from typing import Tuple, Dict, Any
from agent_base import BaseAgent, AgentOutput, AgentInput

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    中兴捧月 GUI Agent 参赛类 - 基线试水版
    核心目标：确保 API 调用成功，并使用正则安全解析坐标和动作，防止任何异常导致计分失败。
    """

    def _initialize(self):
        """初始化内部状态"""
        self.step_history_text = []

    def reset(self):
        """每个测试用例开始前重置状态"""
        self.step_history_text.clear()

        def _build_system_prompt(self, instruction: str) -> str:
            """
            重写系统提示词，进一步强调输出格式。
            """
            return f"""You are an intelligent GUI Agent for mobile devices. 
    Your task is to help the user complete the following instruction:
    【{instruction}】

    Please observe the provided mobile screen screenshot. The screen coordinates are normalized to a 1000x1000 grid (x: 0-1000 from left to right, y: 0-1000 from top to bottom).

    You must output your thinking process in the "Thought" section, and your final action in the "Action" section.
    For CLICK actions, provide a single center point [x, y]. DO NOT output bounding boxes like [x1, y1, x2, y2].

    The Action MUST strictly match one of the following exact formats:
    - CLICK:[[x, y]]
    - TYPE:['text content']
    - SCROLL:[[x1, y1], [x2, y2]]
    - OPEN:['app name']
    - COMPLETE:[]

    Example output:
    Thought: 我需要点击屏幕底部的“我的”标签，中心坐标大约在 x=800, y=950。
    Action: CLICK:[[800, 950]]
    """

    def act(self, input_data: AgentInput) -> AgentOutput:
        """
        Agent 核心方法：根据输入生成动作
        """
        # 1. 生成包含历史消息的 messages
        messages = self.generate_messages(input_data)

        try:
            # 2. 调用大模型 (使用温度 0.1 保证输出的确定性)
            response = self._call_api(messages, temperature=0.1)
            raw_output = response.choices[0].message.content
            usage_info = self.extract_usage_info(response)

            # 3. 使用鲁棒的正则表达式安全提取 action 和 parameters
            action, params = self._robust_parse(raw_output)

            # 4. 记录纯文本历史（为后续多轮记忆做准备）
            self.step_history_text.append(f"Action taken: {action}, Params: {params}")

            return AgentOutput(
                action=action,
                parameters=params,
                raw_output=raw_output,
                usage=usage_info
            )

        except Exception as e:
            logger.error(f"[Agent Error] API调用或解析发生异常: {e}")
            # 【安全兜底策略】如果发生任何异常，返回合法的空操作（这里用点击左上角盲区模拟）或者 COMPLETE，
            # 绝对不要让 Agent 直接抛出异常，防止 test_runner 崩溃导致 0 分。
            return AgentOutput(
                action="CLICK",
                parameters={"point": [0, 0]},
                raw_output=f"Error Handled: {str(e)}"
            )

    def _robust_parse(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        升级版鲁棒解析器：支持自动将 Bounding Box (4坐标) 转换为中心点 (2坐标)
        并严格隔离 Thought 区域，防止误抓取数字。
        """
        # 1. 隔离干扰：只截取 Action: 后面的部分进行解析
        action_text = raw_text
        if "Action:" in raw_text:
            action_text = raw_text.split("Action:")[-1].strip()

        # 2. 增强型 CLICK 匹配（处理 GLM 喜欢输出 [x1, y1, x2, y2] 的习惯）
        # 先尝试匹配 4 个数字的 Bounding Box
        match_click_4 = re.search(r'CLICK:\s*\[?\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]?', action_text)
        if match_click_4:
            # 如果输出的是边界框，自动计算中心点！
            x1, y1, x2, y2 = map(int, match_click_4.groups())
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            logger.info(f"✨ 自动将边界框 [{x1},{y1},{x2},{y2}] 转换为中心点 [{center_x},{center_y}]")
            return "CLICK", {"point": [center_x, center_y]}

        # 再尝试匹配标准的 2 个数字的 Point
        match_click_2 = re.search(r'CLICK:\s*\[?\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_click_2:
            return "CLICK", {"point": [int(match_click_2.group(1)), int(match_click_2.group(2))]}

        # 3. 匹配 SCROLL:[[x1, y1], [x2, y2]]
        match_scroll = re.search(r'SCROLL:\s*\[?\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_scroll:
            return "SCROLL", {
                "start_point": [int(match_scroll.group(1)), int(match_scroll.group(2))],
                "end_point": [int(match_scroll.group(3)), int(match_scroll.group(4))]
            }

        # 4. 匹配 TYPE:['内容'] 或 TYPE:["内容"]
        match_type = re.search(r'TYPE:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_type:
            return "TYPE", {"text": match_type.group(1)}

        # 5. 匹配 OPEN:['应用名']
        match_open = re.search(r'OPEN:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_open:
            return "OPEN", {"app_name": match_open.group(1)}

        # 6. 匹配 COMPLETE:[]
        if 'COMPLETE' in action_text:
            return "COMPLETE", {}

        # 7. 终极兜底：如果还是没匹配上，只在 action_text (排除Thought) 中找数字
        logger.warning(f"未能严格正则匹配，尝试从 Action 区域模糊推断: {action_text}")
        if "CLICK" in action_text:
            nums = re.findall(r'\d+', action_text)
            if len(nums) >= 2:
                return "CLICK", {"point": [int(nums[0]), int(nums[1])]}

        # 最坏的情况，返回安全动作
        return "COMPLETE", {}
