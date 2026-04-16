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
                # 深度死锁：连续多次无效，下达最后通牒
                deadlock_warning = (
                    "\n🔥🔥🔥【系统最高级别警告：深度死锁】🔥🔥🔥\n"
                    "你已经连续多次在同一个无响应页面上浪费操作！\n"
                    "立即停止点击当前区域！如果你找不到目标，必须执行 SCROLL 大范围滑动屏幕寻找，或者点击页面上的【返回/关闭】按钮退出当前死胡同！\n"
                    "如果再次重复相同动作，任务将被判定失败！\n\n"
                )
                logger.error("检测到深度画面死锁，已向大模型注入最高级别纠偏警告！")
            else:
                # 浅层死锁：首次无效
                deadlock_warning = (
                    "\n🔴【系统强制警告】：你上一步执行的动作没有产生任何效果！屏幕画面没有任何变化。\n"
                    "说明上一步的坐标无效。这一次【绝对禁止】点击或滑动之前的相同坐标！请寻找其他 UI 元素或执行 SCROLL 探索！\n\n"
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