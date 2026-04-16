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
    is_submit_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜", "评论", "发布", "发送", "发", "留言", "回复", "输入"])

    plan_reminder = (
        f"\n💡 【任务进度纠偏】：当前全局任务是【{instruction}】。请立即回顾上方的 [当前任务计划]！"
        "认真思考：你现在正处于计划的哪个阶段？还缺哪几个关键分步没有完成？请放弃刚才的无效动作，重点聚焦在【尚未完成的 plan 分步】上，寻找画面中真正能推进进度的 UI 入口！"
    )

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
        if "终态验证" not in raw_output:
            return {
                # 拼接计划唤醒词
                "reviewer_feedback": f"REJECT: 🚨 任务尚未完成！你尝试输出 COMPLETE 提前结束任务。{plan_reminder}",
                "retry_count": retry_count + 1,
            }

    if action == "COMPLETE" and is_submit_task and not has_typed:
        return {
            "reviewer_feedback": f"REJECT: 搜索任务尚未完成关键词输入与确认，严禁提前 COMPLETE！{plan_reminder}",
            "retry_count": retry_count + 1,
        }

    if last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": f"REJECT: 连续执行 TYPE。请优先寻找页面上的原生确认/搜索按钮执行 CLICK。{plan_reminder}",
            "retry_count": retry_count + 1,
        }

    if ("地图" in instruction or "百度地图" in instruction) and last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": f"REJECT: 地图双输入约束触发。请先点击终点输入入口，再输入终点。{plan_reminder}",
            "retry_count": retry_count + 1,
        }

    if action == "COMPLETE" and last_action == "TYPE":
        return {
            "reviewer_feedback": f"REJECT: 刚执行完 TYPE 输入，尚未点击搜索确认或进入详情页，不能直接 COMPLETE。{plan_reminder}",
            "retry_count": retry_count + 1,
        }

    if action == "CLICK" and last_action == "TYPE":
        point = params.get("point", [])
        if len(point) == 2 and point[1] > 600:
            return {
                "reviewer_feedback": f"REJECT: 🚨 键盘禁区触发！严禁点击底部系统键盘区域！请去页面上方寻找原生搜索按钮。{plan_reminder}",
                "retry_count": retry_count + 1,
            }

    if action == "TYPE" and last_action not in ["CLICK", "OPEN"]:
        return {
            "reviewer_feedback": f"REJECT: 🚨 输入框激活悖论触发！你必须先 CLICK 点击搜索框激活光标，才能 TYPE。{plan_reminder}",
            "retry_count": retry_count + 1,
        }

    return {"reviewer_feedback": "PASS"}