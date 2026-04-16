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

    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()

    instruction = input_data.instruction or ""
    is_search_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜"])

    # 检查历史动作中是否已经包含过 TYPE（输入动作）
    has_typed = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )

    # ---------------------------------------------------------
    # 纯代码规则强拦截器 (0耗时)
    # ---------------------------------------------------------

    # 规则 1: 搜索任务强阻断 - 必须先输入关键词，才能 COMPLETE
    if action == "COMPLETE" and is_search_task and not has_typed:
        return {
            "reviewer_feedback": "REJECT: 搜索任务尚未完成关键词输入与搜索确认，不能提前 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    # 规则 2: 防止连续盲目输入 - 连续两步执行 TYPE 动作大概率是错的，要求先进行确认/点击
    if last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 连续执行 TYPE。请优先 ENTER 或点击搜索/确认按钮。",
            "retry_count": retry_count + 1,
        }

    # 规则 3: 地图类任务双输入约束 - 避免起点和终点重叠输入
    if ("地图" in instruction or "百度地图" in instruction) and last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 地图双输入约束触发。请先点击终点输入入口，再输入终点。",
            "retry_count": retry_count + 1,
        }

    # 规则 4: 逻辑阻断 - 输入后直接 COMPLETE 通常是典型的错误（通常需要先点搜索，再点目标内容）
    if action == "COMPLETE" and last_action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 刚执行完 TYPE 输入，尚未点击搜索确认或查看结果，不能直接 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    # ---------------------------------------------------------
    # 规则全部通过，直接放行 (省去了原本耗时极长的大模型 API 调用)
    # ---------------------------------------------------------
    return {"reviewer_feedback": "PASS"}