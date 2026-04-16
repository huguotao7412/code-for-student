import logging
from typing import Any, Dict

from agent_base import AgentOutput
from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)

def format_output_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    proposed_action = state.get("proposed_action", "CLICK")
    proposed_params = state.get("proposed_params", {"point": [500, 500]})
    feedback = state.get("reviewer_feedback", "")

    # 【最终拦截逻辑】如果带着 REJECT 进来，说明重试 3 次耗尽模型依然固执己见
    if "REJECT" in feedback:
        logger.error(f"模型重试耗尽，拦截毒药动作并执行边缘无害化点击")
        proposed_action = "CLICK"
        # 【修改点】：不再点屏幕中心 [500,500]，改为点屏幕最左上角 [10, 10] (状态栏白区)
        # 这是一个无害动作，既不中断任务，也不引发误触，让其在下个循环再做打算。
        proposed_params = {"point": [10, 10]}

    action, params, expected_effect = agent._normalize_output(
        proposed_action,
        proposed_params,
    )

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