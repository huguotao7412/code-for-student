# src/utils/state.py
from typing import List

class GUIState:
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self.step_count = 0
        self.action_history: List[str] = []
        self.global_plan: str = ""
        self.last_expected_effect: str = "无（这是第一步）"
        self.stuck_level = 0

    def update(self, action_str: str, expected_effect: str = ""):
        """记录格式化后的动作，并更新步数"""
        self.action_history.append(action_str)
        self.step_count += 1

        if expected_effect:
            self.last_expected_effect = expected_effect

        # 基于重复动作提升卡死等级，基于变化逐步恢复
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            self.stuck_level = min(self.stuck_level + 1, 3)
        elif len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            self.stuck_level = max(self.stuck_level, 1)
        else:
            self.stuck_level = max(self.stuck_level - 1, 0)

    def is_stuck(self) -> bool:
        """检测是否陷入死循环：如果最近3步动作完全相同，则判定为卡死"""
        if len(self.action_history) >= 3:
            return len(set(self.action_history[-3:])) == 1
        return False

    def should_force_stop(self) -> bool:
        """检查是否触达步数上限"""
        return self.step_count >= self.max_steps

    def get_summary(self) -> str:
        """生成精简的历史摘要（状态裁剪），仅保留最近 5 步防 Token 溢出"""
        if not self.action_history:
            return "None"
        return " -> ".join(self.action_history[-5:]) + f" | last_effect={self.last_expected_effect}"
