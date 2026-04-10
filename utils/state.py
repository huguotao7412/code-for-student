# src/utils/state.py
from typing import List, Any, Optional

class GUIState:
    def __init__(self, max_steps: int = 15):
        self.max_steps = max_steps
        self.step_count = 0
        self.action_history: List[str] = []
        self.global_plan: str = ""
        self.last_expected_effect: str = "无（这是第一步）"
        self.stuck_level = 0
        self.last_visual_hash: str = ""
        self.last_visual_changed: bool = True
        self.visual_repeat_count: int = 0
        self.last_image: Optional[Any] = None
        self.macro_stage: str = ""
        self.macro_text: str = ""
        self.macro_target: str = ""

    def update(self, action_str: str, expected_effect: str = "", visual_hash: str = "", visual_changed: Optional[bool] = None):
        """记录格式化后的动作，并更新步数；可选记录视觉变化信号。"""
        self.action_history.append(action_str)
        self.step_count += 1

        if expected_effect:
            self.last_expected_effect = expected_effect

        if visual_hash:
            if self.last_visual_hash and visual_hash == self.last_visual_hash:
                self.visual_repeat_count = min(self.visual_repeat_count + 1, 3)
                self.last_visual_changed = False
            else:
                self.visual_repeat_count = max(self.visual_repeat_count - 1, 0)
                self.last_visual_changed = True
            self.last_visual_hash = visual_hash
        elif visual_changed is not None:
            self.last_visual_changed = visual_changed
            if visual_changed:
                self.visual_repeat_count = max(self.visual_repeat_count - 1, 0)
            else:
                self.visual_repeat_count = min(self.visual_repeat_count + 1, 3)

        # 基于重复动作提升卡死等级，基于变化逐步恢复
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            self.stuck_level = min(self.stuck_level + 1, 3)
        elif len(self.action_history) >= 2 and self.action_history[-1] == self.action_history[-2]:
            self.stuck_level = max(self.stuck_level, 1)
        else:
            self.stuck_level = max(self.stuck_level - 1, 0)

        if self.visual_repeat_count >= 2:
            self.stuck_level = min(self.stuck_level + 1, 3)

    def is_stuck(self) -> bool:
        """检测是否陷入死循环：如果最近3步动作完全相同，则判定为卡死"""
        if len(self.action_history) >= 3:
            return len(set(self.action_history[-3:])) == 1
        return False

    def should_force_stop(self) -> bool:
        """检查是否触达步数上限"""
        return self.step_count >= self.max_steps

    def get_recent_actions(self, window: int = 2) -> List[str]:
        """返回最近若干步的动作字符串，便于短窗决策。"""
        if not self.action_history:
            return []
        if window <= 0:
            return []
        return self.action_history[-window:]

    def get_summary(self) -> str:
        """生成精简的历史摘要：只保留最近 2 步，避免长历史干扰当前决策。"""
        if not self.action_history:
            return "None"
        recent = self.action_history[-2:]
        macro = f" | macro={self.macro_stage}:{self.macro_text}" if self.macro_stage else ""
        visual = f" | visual_repeat={self.visual_repeat_count}, changed={self.last_visual_changed}"
        return " -> ".join(recent) + f" | last_effect={self.last_expected_effect}{visual}{macro}"
