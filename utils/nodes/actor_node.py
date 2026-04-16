import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def actor_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    feedback = state.get("reviewer_feedback", "")
    history_actions = input_data.history_actions or []

    # --------------------------------------------------------------------
    # Hash 死锁破局检测核心逻辑 (升级版)
    # --------------------------------------------------------------------
    curr_hash = agent._image_signature(input_data.current_image)
    last_hash = getattr(agent, "_prev_image_hash", None)

    # 利用状态机中的 visual_repeat_count 来判断死锁深度
    repeat_count = agent.state.visual_repeat_count

    deadlock_warning = ""
    if history_actions:
        last_action_dict = history_actions[-1] if isinstance(history_actions[-1], dict) else {}
        last_action = str(last_action_dict.get("action", "")).upper()

        if last_action in ["CLICK", "SCROLL"] and last_hash and curr_hash == last_hash:
            if repeat_count >= 2:
                # 针对离线评测的深度死锁警告
                deadlock_warning = (
                    "\n🔥🔥🔥【系统最高级别警告：坐标连续偏离】🔥🔥🔥\n"
                    "你已经连续多次点击了无效区域！当前任务的关键入口【绝对就在当前屏幕上】！\n"
                    "立即停止点击刚才的位置！请仔细重新审视画面，寻找你遗漏的、更符合任务目标的按钮或文本区域。\n"
                    "【严禁】尝试 SCROLL 滑动或点击返回逃避，必须在当前页面找到正确入口！\n\n"
                )
            else:
                deadlock_warning = (
                    "\n🔴【系统强制警告】：你上一步的动作无效，画面未更新。\n"
                    "说明你上一步输出的坐标偏离了目标控件，或者该区域不可交互。这一次【绝对禁止】使用相同的坐标！请微调坐标或寻找真正的控件区域！\n\n"
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