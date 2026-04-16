import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def actor_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    feedback = state.get("reviewer_feedback", "")
    history_actions = input_data.history_actions or []

    # --------------------------------------------------------------------
    # Hash 死锁破局检测核心逻辑
    # --------------------------------------------------------------------
    curr_hash = agent._image_signature(input_data.current_image)
    last_hash = getattr(agent, "_prev_image_hash", None)

    deadlock_warning = ""
    if history_actions:
        last_action_dict = history_actions[-1] if isinstance(history_actions[-1], dict) else {}
        last_action = str(last_action_dict.get("action", "")).upper()

        if last_action in ["CLICK", "SCROLL"] and last_hash and curr_hash == last_hash:
            deadlock_warning = (
                "\n🔴【系统强制警告】：你上一步执行的动作没有产生任何效果！屏幕画面没有任何变化。\n"
                "这说明你上一步选择的坐标是无效的空白区域或发生了误触。\n"
                "这一次【绝对禁止】点击或滑动之前的相同坐标！请仔细寻找页面上其他可交互的元素，或者尝试执行 SCROLL 探索新区域！\n"
            )
            logger.warning("检测到画面死锁，已向大模型注入强制纠偏警告！")

    agent._prev_image_hash = curr_hash
    # --------------------------------------------------------------------

    prompt = agent._build_prompt(
        input_data.instruction,
        input_data.history_actions,
        reviewer_feedback=feedback,
        retry_count=state.get("retry_count", 0),
    )

    if deadlock_warning:
        prompt = deadlock_warning + prompt

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": agent._encode_image(input_data.current_image)}},
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
        "proposed_action": model_action,
        "proposed_params": model_params,
        "model_effect": model_effect,
        "raw_output": raw_output,
        "usage": usage,
    }