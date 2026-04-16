import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action = str(state.get("proposed_action", "")).upper()
    params = state.get("proposed_params", {}) or {}
    input_data = state["input_data"]
    history_actions = state.get("history_actions") or input_data.history_actions or []
    retry_count = int(state.get("retry_count", 0))
    raw_output = str(state.get("raw_output", ""))

    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()

    instruction = input_data.instruction or ""
    is_search_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜"])

    has_typed = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )

    # ---------------------------------------------------------
    # 纯代码规则强拦截器 (0耗时，防早退)
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 纯代码规则强拦截器 (0耗时，防早退)
    # ---------------------------------------------------------

    if action == "COMPLETE":
        # 只要没有提及“终态验证”，一律打回
        if "终态验证" not in raw_output:
            return {
                "reviewer_feedback": "REJECT: 你尝试输出 COMPLETE，但你声明的当前状态不是 [State: 终态验证]。必须进入最终结果页面（如视频播放页、商品详情页）才能 COMPLETE。",
                "retry_count": retry_count + 1,
            }

    if action == "COMPLETE" and is_search_task and not has_typed:
        return {
            "reviewer_feedback": "REJECT: 搜索任务尚未完成关键词输入与搜索确认，不能提前 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    if last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 连续执行 TYPE。请优先 ENTER 或点击搜索/确认按钮。",
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

    return {"reviewer_feedback": "PASS"}