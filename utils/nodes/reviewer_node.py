import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action = str(state.get("proposed_action", "")).upper()
    params = state.get("proposed_params", {}) or {}
    input_data = state["input_data"]

    # ---------------------------------------------------------
    # 【新增 P3：双重历史提取】防止外部评测机传回空历史
    # ---------------------------------------------------------
    history_actions = state.get("history_actions") or input_data.history_actions or []
    internal_history = agent.state.action_history or []

    retry_count = int(state.get("retry_count", 0))
    raw_output = str(state.get("raw_output", ""))

    # 提取最后一次动作
    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()
    elif internal_history:
        # 如果外部丢失，从内部格式 "ACTION:{...}" 中提取
        last_action = internal_history[-1].split(":")[0].upper()

    instruction = input_data.instruction or ""
    is_search_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜"])

    # 判定是否执行过输入（双重保险）
    has_typed_external = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )
    has_typed_internal = any("TYPE" in str(act).upper() for act in internal_history)
    has_typed = has_typed_external or has_typed_internal

    # ---------------------------------------------------------
    # 纯代码规则强拦截器 (0耗时，防早退)
    # ---------------------------------------------------------

    if action == "COMPLETE":
        # 放宽匹配条件，只要包含"终态验证"即可通过，防范缺少括号的幻觉
        if "终态验证" not in raw_output:
            return {
                "reviewer_feedback": "REJECT: 你尝试输出 COMPLETE，但你声明的当前状态不是 [State: 终态验证]。必须进入最终页面才能 COMPLETE。",
                "retry_count": retry_count + 1,
            }

    if action == "COMPLETE" and is_search_task and not has_typed:
        return {
            "reviewer_feedback": "REJECT: 搜索任务尚未完成关键词输入与搜索确认，不能提前 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    if last_action == "TYPE" and action == "TYPE":
        # 移除 ENTER 提示，引导模型去点击按钮
        return {
            "reviewer_feedback": "REJECT: 连续执行 TYPE。请优先寻找页面上的原生确认/搜索按钮执行 CLICK。",
            "retry_count": retry_count + 1,
        }

    if ("地图" in instruction or "百度地图" in instruction) and last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 地图双输入约束触发。请先点击终点输入入口，再输入终点。",
            "retry_count": retry_count + 1,
        }

    if action == "COMPLETE" and last_action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 刚执行完 TYPE 输入，尚未点击搜索确认或进入详情页，不能直接 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    if action == "CLICK" and last_action == "TYPE":
        point = params.get("point", [])
        # 假设 1000x1000 的归一化坐标体系，y > 600 通常是虚拟键盘弹出的覆盖区
        if len(point) == 2 and point[1] > 600:
            return {
                "reviewer_feedback": f"REJECT: 🚨 键盘禁区触发！你尝试点击的坐标 {point} 位于底部系统键盘区域。严禁点击系统键盘！请重新观察画面，寻找并点击页面上方的原生【搜索/确认】UI按钮。",
                "retry_count": retry_count + 1,
            }

    return {"reviewer_feedback": "PASS"}