import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def actor_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    feedback = state.get("reviewer_feedback", "")
    history_actions = input_data.history_actions or []

    # --------------------------------------------------------------------
    # 核心增强 1：视觉反思 (Expected vs. Reality)
    # --------------------------------------------------------------------
    reflection_warning = ""
    last_expectation = getattr(agent, "_last_expected_effect", None)

    if history_actions and last_expectation:
        reflection_warning = (
            f"\n🔍【视觉反思（非常重要）】：\n"
            f"你上一步的操作预期是：【{last_expectation}】。\n"
            f"请你对比【当前最新截图】和你的预期：预期是否真正达成？\n"
            f"- 如果预期未达成（如没看到弹窗、页面没跳转、搜索没结果），说明上一步点偏了或位置错误！\n"
            f"- 你必须立即在 [Analyze] 中指出失败原因，并修正坐标或策略，绝对禁止重复之前的错误路径！\n"
        )
        logger.info(f"已注入视觉反思逻辑，预期为: {last_expectation}")

    # --------------------------------------------------------------------
    # 核心增强 2：Hash 死锁破局 (Visual Deadlock Breaker)
    # --------------------------------------------------------------------
    curr_hash = agent._image_signature(input_data.current_image)
    last_hash = getattr(agent, "_prev_image_hash", None)

    deadlock_warning = ""
    if history_actions:
        last_action_dict = history_actions[-1] if isinstance(history_actions[-1], dict) else {}
        last_action = str(last_action_dict.get("action", "")).upper()

        if last_action in ["CLICK", "SCROLL"] and last_hash and curr_hash == last_hash:
            deadlock_warning = (
                "\n🔴【系统死锁警告】：画面完全没变！你上一步的动作无效。\n"
                "禁止再次点击相同的坐标！请寻找页面上其他按钮，或尝试 SCROLL 探索。\n"
            )
            logger.warning("检测到画面死锁，已注入强制纠偏警告！")

    agent._prev_image_hash = curr_hash
    # --------------------------------------------------------------------

    # 生成基础 Prompt
    prompt = agent._build_prompt(
        input_data.instruction,
        input_data.history_actions,
        reviewer_feedback=feedback,
        retry_count=state.get("retry_count", 0),
    )

    # 组合 Prompt：反思 > 死锁 > 基础指令
    full_prompt = ""
    if reflection_warning: full_prompt += reflection_warning
    if deadlock_warning: full_prompt += deadlock_warning
    full_prompt += prompt

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
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