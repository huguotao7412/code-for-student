import logging
from typing import Any, Dict

from agent_base import AgentOutput
from utils.graph_state import WorkflowState
from utils.action_sandbox import sanitize_and_stick

logger = logging.getLogger(__name__)


def format_output_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    proposed_action = state.get("proposed_action", "CLICK")
    proposed_params = state.get("proposed_params", {"point": [500, 500]})
    feedback = state.get("reviewer_feedback", "")

    # 【拦截逻辑】如果带着 REJECT 进来，说明重试耗尽且模型固执己见
    if "REJECT" in feedback:
        logger.error(f"模型重试耗尽，强制拦截毒药动作: {proposed_action}:{proposed_params}")
        proposed_action = "CLICK"
        # 兜底：点屏幕正中央或左上角，至少保证是一个合法的 CLICK 格式，避免评测脚本 KeyError
        proposed_params = {"point": [500, 500]}

    # 1. 基础的动作标准化与边界裁剪
    action, params, expected_effect = agent._normalize_output(
        proposed_action,
        proposed_params,
    )

    # 【新增 P2：坐标精度纠偏】将归一化后的坐标通过 Sandbox 吸附到真实的 UI 元素中心
    element_map = getattr(agent, "_current_element_map", {})
    action, params = sanitize_and_stick(action, params, element_map)

    model_effect = state.get("model_effect", "")
    if model_effect and not expected_effect:
        expected_effect = model_effect

    final_output = AgentOutput(
        action=action,
        parameters=params,
        raw_output=state.get("raw_output", ""),
        usage=state.get("usage"),
    )

    return {
        "normalized_action": action,
        "normalized_params": params,
        "expected_effect": expected_effect,
        "final_output": final_output,
    }