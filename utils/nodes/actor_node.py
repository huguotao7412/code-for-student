import logging
from typing import Any, Dict

from utils.a2a_protocol import A2AChannels, ensure_mailbox, read_payload, write_packet
from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def actor_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    mailbox = ensure_mailbox(state)
    review_payload = read_payload(state, A2AChannels.REVIEW, default={}) or {}
    feedback = str(review_payload.get("reviewer_feedback", state.get("reviewer_feedback", "")))
    current_image_url = str(state.get("current_image_url", "") or "") or agent._encode_image(input_data.current_image)

    prompt = agent._build_prompt(
        input_data.instruction,
        input_data.history_actions,
        reviewer_feedback=feedback,
        retry_count=state.get("retry_count", 0),
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": current_image_url}},
        ],
    }]

    raw_output = ""
    usage = None
    try:
        response = agent._call_api(messages, temperature=0.0)
        raw_output = response.choices[0].message.content
        usage = agent.extract_usage_info(response)
        model_action, model_params, model_effect = agent._parse_with_effect(raw_output)
    except Exception as e:
        logger.warning(f"Model failed: {e}")
        model_action, model_params, model_effect = "CLICK", {"point": [500, 500]}, "兜底点击"
        raw_output = f"Fallback: {e}"

    write_packet(
        mailbox,
        channel=A2AChannels.ACTION_PROPOSAL,
        sender="actor",
        receiver="reviewer",
        kind="proposal",
        payload={
            "proposed_action": model_action,
            "proposed_params": model_params,
            "model_effect": model_effect,
            "raw_output": raw_output,
            "usage": usage,
        },
    )

    return {
        "mailbox": mailbox,
        "proposed_action": model_action,
        "proposed_params": model_params,
        "model_effect": model_effect,
        "raw_output": raw_output,
        "usage": usage,
    }
