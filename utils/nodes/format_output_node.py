from typing import Any, Dict

from agent_base import AgentOutput
from utils.graph_state import WorkflowState


def format_output_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action, params, expected_effect = agent._normalize_output(
        state.get("proposed_action", "CLICK"),
        state.get("proposed_params", {"point": [500, 500]}),
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

