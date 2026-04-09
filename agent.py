# 文件路径: agent.py
from dotenv import load_dotenv
load_dotenv()
import re
import logging
from typing import Tuple, Dict, Any, List
from dotenv import load_dotenv

from agent_base import BaseAgent, AgentOutput, AgentInput
# 假设 utils.vision_enhancer 已经存在
from utils.vision_enhancer import add_coordinate_grid
from utils.visual_memory import draw_previous_action
from utils.ui_detector import draw_som_labels
from utils.action_sandbox import sanitize_and_stick

load_dotenv()
logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    符合中兴捧月赛题规范的 GUI Agent
    采用无状态设计 + 强 SOP 约束
    """

    def _initialize(self):
        """初始化方法"""
        self._current_element_map = {}

    def reset(self):
        """重置状态：无状态设计，无需重置历史"""
        pass

    # 【修复点 1】补全函数签名，增加 history_actions 参数
    def _build_advanced_prompt(self, instruction: str, history_actions: List[Dict[str, Any]]) -> str:
        """
        构建基于标准操作程序 (SOP) 的强约束 Prompt
        """
        # 【修复点 2】安全地获取 last_action，容错处理
        last_action = history_actions[-1] if (history_actions and len(history_actions) > 0) else "None"

        return f"""你是一个高级 Android GUI 智能代理。目标：【{instruction}】

        [视觉说明]
        1. 截图已叠加 10x10 红线网格（坐标 0-1000）。
        2. 黄色方框及数字是识别出的 UI 元素编号。
        3. 绿色/蓝色标记是你【上一步】的动作位置。

        [重要：自我反思复盘]
        你上一步执行了：{last_action}。
        请对比当前截图：
        - 如果页面没变化，说明上一步点击无效或未加载，请尝试点击不同位置或重新点击。
        - 如果目标结果已在屏幕呈现，必须立即输出 COMPLETE:[]。

        [操作 SOP]
        1. TYPE 前必须 CLICK 输入框弹出键盘。
        2. 优先点击黄色编号框的中心坐标。

        Thought:
        1. 状态复盘：上一步动作是否生效？
        2. 元素定位：目标元素编号是多少？坐标是多少？
        3. 下一步动作：基于 SOP 应执行什么？
        Action: <格式严格遵循 CLICK:[[x,y]] / TYPE:['text'] / SCROLL:[[x,y],[x,y]] / OPEN:['app'] / COMPLETE:[]>
        """

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """覆盖基类的消息生成方法"""
        # 1. 元素检测与编号 (SoM)
        img, self._current_element_map = draw_som_labels(input_data.current_image)

        # 2. 绘制上一步视觉记忆
        img = draw_previous_action(img, input_data.history_actions)

        # 3. 叠加坐标网格
        enhanced_image = add_coordinate_grid(img)

        # 4. 构建带反思内容的 Prompt (此时参数对应已正确)
        final_prompt = self._build_advanced_prompt(input_data.instruction, input_data.history_actions)

        return [
            {"role": "user", "content": [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": self._encode_image(enhanced_image)}}
            ]}
        ]

    def act(self, input_data: AgentInput) -> AgentOutput:
        """核心动作执行方法，符合基类调用规范"""
        messages = self.generate_messages(input_data)
        try:
            response = self._call_api(messages, temperature=0.0)
            raw_output = response.choices[0].message.content
            usage_info = self.extract_usage_info(response)

            action, params = self._robust_parse(raw_output)

            # 执行沙盒吸附修正
            action, params = sanitize_and_stick(action, params, self._current_element_map)

            return AgentOutput(action=action, parameters=params, raw_output=raw_output, usage=usage_info)

        # 【修复点 3】删除了重复的 except Exception 块，保留一个规范的错误兜底
        except Exception as e:
            logger.error(f"[Agent Error]: {e}")
            # 安全兜底：如果 API 报错或解析彻底失败，默认点击中心点，防止脚本崩溃中断测试
            return AgentOutput(
                action="CLICK",
                parameters={"point": [500, 500]},
                raw_output=f"Error occurred: {str(e)}"
            )

    def _robust_parse(self, raw_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        强壮的解析器，确保输出完全符合基类要求的常量格式
        """
        # 提取 Action 后的文本，并去除多余空白
        action_text = raw_text.split("Action:")[-1].strip() if "Action:" in raw_text else raw_text.strip()

        if 'COMPLETE' in action_text:
            return "COMPLETE", {}

        match_click = re.search(r'CLICK:\s*\[?\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_click:
            return "CLICK", {"point": [int(match_click.group(1)), int(match_click.group(2))]}

        match_scroll = re.search(r'SCROLL:\s*\[?\[(\d+),\s*(\d+)\],\s*\[(\d+),\s*(\d+)\]\]?', action_text)
        if match_scroll:
            return "SCROLL", {
                "start_point": [int(match_scroll.group(1)), int(match_scroll.group(2))],
                "end_point": [int(match_scroll.group(3)), int(match_scroll.group(4))]
            }

        # 【修改点 1】：兼容 TYPE 缺失括号、单双引号混用、甚至中文引号的情况
        # 能够成功匹配: TYPE:['内容'], TYPE:内容, TYPE:"内容", TYPE:['内容'] 等
        match_type = re.search(r'TYPE:\s*\[?[\'\"‘“]?(.*?)[\'\"’”]?\]?$', action_text)
        if match_type:
            # .strip() 用于去除模型可能额外输出的前后空格
            return "TYPE", {"text": match_type.group(1).strip()}

        # 【修改点 2】：对 OPEN 也做同样的鲁棒性兼容
        match_open = re.search(r'OPEN:\s*\[?[\'\"‘“]?(.*?)[\'\"’”]?\]?$', action_text)
        if match_open:
            return "OPEN", {"app_name": match_open.group(1).strip()}

        # 最终兜底：防止程序崩溃
        return "CLICK", {"point": [500, 500]}