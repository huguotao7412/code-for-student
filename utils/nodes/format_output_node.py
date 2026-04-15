from typing import Any, Dict

from agent_base import AgentOutput
from utils.a2a_protocol import A2AChannels, ensure_mailbox, read_payload, write_packet
from utils.graph_state import WorkflowState


def format_output_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    mailbox = ensure_mailbox(state)
    proposal = read_payload(state, A2AChannels.ACTION_PROPOSAL, default={}) or {}

    action, params, expected_effect = agent._normalize_output(
        proposal.get("proposed_action", state.get("proposed_action", "CLICK")),
        proposal.get("proposed_params", state.get("proposed_params", {"point": [500, 500]})),
    )

    model_effect = proposal.get("model_effect", state.get("model_effect", ""))
    if model_effect and not expected_effect:
        expected_effect = model_effect

    final_output = AgentOutput(
        action=action,
        parameters=params,
        raw_output=proposal.get("raw_output", state.get("raw_output", "")),
        usage=proposal.get("usage", state.get("usage")),
    )

    write_packet(
        mailbox,
        channel=A2AChannels.FINAL_OUTPUT,
        sender="format_output",
        receiver="orchestrator",
        kind="final_output",
        payload={
            "normalized_action": action,
            "normalized_params": params,
            "expected_effect": expected_effect,
            "final_output": final_output,
        },
    )

    return {
        "mailbox": mailbox,
        "normalized_action": action,
        "normalized_params": params,
        "expected_effect": expected_effect,
        "final_output": final_output,
    }
