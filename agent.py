# 文件路径: agent.py
from dotenv import load_dotenv
load_dotenv()
# 文件路径: src/agent.py
import re
import logging
from typing import Tuple, Dict, Any, List
from agent_base import BaseAgent, AgentOutput, AgentInput
# 假设 utils.vision_enhancer 已经存在，用于给图片画网格
from utils.vision_enhancer import add_coordinate_grid

logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """
    符合中兴捧月赛题规范的 GUI Agent
    采用无状态设计 + 强 SOP 约束
    """

    def _initialize(self):
        """初始化方法"""
        pass

    def reset(self):
        """重置状态：无状态设计，无需重置历史"""
        pass

    def _build_advanced_prompt(self, instruction: str) -> str:
        """
        构建基于标准操作程序 (SOP) 的强约束 Prompt
        """
        return f"""你是一个高级 Android 设备 GUI 智能代理。
你的最终目标是完成任务：【{instruction}】

[屏幕与坐标系信息]
提供的截图已叠加了相对坐标系网格。整个屏幕的坐标范围是 X:[0, 1000], Y:[0, 1000]。
左上角为 [0, 0]，右下角为 [1000, 1000]。请利用网格粗略定位，然后结合 UI 元素中心点估算精确坐标。

[Android 操作 SOP - 必须严格遵守]
1. 【输入文本规范】：
   - 绝不能在软键盘未弹出时直接使用 TYPE 动作。
   - 如果你需要输入文字，请先检查屏幕底部是否有软键盘。如果没有，你的动作必须是 CLICK 点击目标输入框。
   - 只有当软键盘已经显示在屏幕上，或者当前焦点明确在输入框内时，才能使用 TYPE 动作。
2. 【确认与搜索规范】：
   - 输入文字后，若需要触发搜索或确认，请 CLICK 键盘右下角的“搜索/回车”按钮，或页面上的“搜索”文本，不要试图寻找不存在的 ENTER 动作。
3. 【任务完成规范】：
   - 仔细审视当前截图，如果任务所要求的结果（例如：视频已开始播放、评论已发布、路线已规划、页面已打开）在当前截图中【已经明确呈现】，你必须立即输出 COMPLETE:[]。不要做任何多余操作。
4. 【滚动规范】：
   - 如果需要向下浏览内容，请执行向上滑动，格式推荐为：SCROLL:[[500, 800], [500, 200]]。

[输出格式限制]
请严格按照以下结构输出，先进行 Thought 分析，最后给出 Action：

Thought:
1. 任务进度：当前页面处于任务的哪个阶段？
2. 视觉分析：页面上关键 UI 元素在哪？软键盘处于什么状态？
3. 下一步动作：基于上述 SOP，我当前这一步应该执行什么动作？
4. 坐标计算：如果需要 CLICK 或 SCROLL，目标中心坐标 X 和 Y 分别是多少？
Action: <仅限下方5种格式之一>

合法的 Action 格式（必须严格匹配，不要随意编造）：
CLICK:[[x, y]]
TYPE:['需要输入的文本']
SCROLL:[[start_x, start_y], [end_x, end_y]]
OPEN:['应用名称']
COMPLETE:[]
"""

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """覆盖基类的消息生成方法"""
        # 给图片添加网格（此步骤极大地提升了模型预测坐标的准确率）
        enhanced_image = add_coordinate_grid(input_data.current_image)

        # 摒弃历史消息，使用无状态的强约束 Prompt
        final_prompt = self._build_advanced_prompt(input_data.instruction)

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
            # 使用基类受保护的方法调用 API，严格遵循规则，不传入敏感 kwargs
            # temperature 设为 0.0 以获得稳定的格式输出
            response = self._call_api(messages, temperature=0.0)
            raw_output = response.choices[0].message.content
            usage_info = self.extract_usage_info(response)

            action, params = self._robust_parse(raw_output)

            return AgentOutput(
                action=action,
                parameters=params,
                raw_output=raw_output,
                usage=usage_info
            )

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
        # 提取 Action 后的文本
        action_text = raw_text.split("Action:")[-1].strip() if "Action:" in raw_text else raw_text

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

        match_type = re.search(r'TYPE:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_type:
            return "TYPE", {"text": match_type.group(1)}

        match_open = re.search(r'OPEN:\s*\[[\'"](.*?)[\'"]\]', action_text)
        if match_open:
            return "OPEN", {"app_name": match_open.group(1)}

        # 最终兜底
        return "CLICK", {"point": [500, 500]}