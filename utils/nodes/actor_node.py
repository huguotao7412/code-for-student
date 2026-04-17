import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState
from utils.ui_detector import draw_som_labels

logger = logging.getLogger(__name__)


def actor_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    feedback = state.get("reviewer_feedback", "")

    task_plan = state.get("task_plan") or []
    completed_tasks = state.get("completed_tasks") or []
    app_name = state.get("app_name") or agent._app_name or ""

    active_task = agent._next_pending_task(task_plan, completed_tasks)
    active_task_id = active_task.get("id") if active_task else None

    curr_hash_global = agent._image_signature(input_data.current_image)
    curr_hash_local = agent._focus_signature(input_data.current_image)
    prev_hash_global = state.get("frame_hash_global") or agent.state.last_visual_hash
    prev_hash_local = state.get("frame_hash_local") or agent.state.last_visual_hash_local
    hash_changed = (
        not prev_hash_global
        or curr_hash_global != prev_hash_global
        or (curr_hash_local and prev_hash_local and curr_hash_local != prev_hash_local)
    )

    som_image = input_data.current_image
    som_map: Dict[int, Dict[str, Any]] = {}
    try:
        som_image, som_map = draw_som_labels(input_data.current_image)
    except Exception as e:
        logger.warning(f"SOM draw failed: {e}")


    prompt = agent._build_prompt(
        input_data.instruction,
        input_data.history_actions,
        reviewer_feedback=feedback,
        retry_count=state.get("retry_count", 0),
        app_name=app_name,
        completed_tasks=completed_tasks,
        active_task=active_task,
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": agent._encode_image(input_data.current_image)}},
            {"type": "image_url", "image_url": {"url": agent._encode_image(som_image)}},
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

    return {
        "active_task_id": active_task_id,
        "som_map": som_map,
        "frame_hash_global": curr_hash_global,
        "frame_hash_local": curr_hash_local,
        "hash_changed": hash_changed,
        "proposed_action": model_action,
        "proposed_params": model_params,
        "model_effect": model_effect,
        "raw_output": raw_output,
        "usage": usage,
    }