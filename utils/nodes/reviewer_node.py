import json
import logging
from typing import Any, Dict, List

from utils.a2a_protocol import A2AChannels, ensure_mailbox, read_payload, write_packet
from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return {}
    return {}


def _count_action(history_actions: List[Dict[str, Any]], action_name: str) -> int:
    target = action_name.upper()
    return sum(
        1
        for item in (history_actions or [])
        if isinstance(item, dict) and str(item.get("action", "")).upper() == target
    )


def _extract_type_text(action: str, params: Dict[str, Any]) -> str:
    if action.upper() != "TYPE" or not isinstance(params, dict):
        return ""
    return str(params.get("text", params.get("content", ""))).strip()


def _review_reject(mailbox: Dict[str, Any], retry_count: int, feedback: str) -> Dict[str, Any]:
    write_packet(
        mailbox,
        channel=A2AChannels.REVIEW,
        sender="reviewer",
        receiver="actor",
        kind="review",
        payload={"reviewer_feedback": feedback, "retry_count": retry_count + 1, "verdict": "REJECT"},
    )
    return {"mailbox": mailbox, "reviewer_feedback": feedback, "retry_count": retry_count + 1}


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    mailbox = ensure_mailbox(state)
    proposal = read_payload(state, A2AChannels.ACTION_PROPOSAL, default={}) or {}

    action = str(proposal.get("proposed_action", state.get("proposed_action", ""))).upper()
    params = proposal.get("proposed_params", state.get("proposed_params", {})) or {}
    input_data = state["input_data"]
    history_actions = state.get("history_actions") or input_data.history_actions or []
    retry_count = int(state.get("retry_count", 0))

    task_type = str(state.get("task_type", "general"))
    task_slots = state.get("task_slots") or {}
    flow_flags = state.get("flow_flags") or {}

    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()

    instruction = input_data.instruction or ""
    is_search_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜"])
    has_typed = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )
    click_count = int(flow_flags.get("click_count", _count_action(history_actions, "CLICK")))
    search_like = is_search_task or task_type in {"search", "food", "map", "flight"}

    # 仅在“明显还处于输入链路且早期就要 COMPLETE”的场景做硬拦截，避免过拟合误拒绝。
    if action == "COMPLETE" and search_like and not has_typed and click_count <= 2 and retry_count == 0:
        return _review_reject(mailbox, retry_count, "REJECT: 输入链路可能尚未完成，请先激活输入框并 TYPE 后再 COMPLETE。")

    if last_action == "TYPE" and action == "TYPE":
        last_params = history_actions[-1].get("parameters", {}) if history_actions and isinstance(history_actions[-1], dict) else {}
        last_text = _extract_type_text("TYPE", last_params)
        current_text = _extract_type_text(action, params)
        if not current_text or current_text == last_text:
            return _review_reject(mailbox, retry_count, "REJECT: 连续执行 TYPE 且文本未变化。请优先 ENTER 或点击搜索/确认按钮。")

    if search_like and not has_typed and action == "CLICK" and click_count >= 3 and retry_count == 0:
        return _review_reject(mailbox, retry_count, "REJECT: 输入链路缺失。请先激活输入框并 TYPE 任务词。")

    prompt = f"""你是移动端 GUI 动作审核员。你只负责判断当前候选动作是否与任务目标和截图证据一致。
仅输出 JSON：
{{
  \"verdict\": \"PASS\" 或 \"REJECT\",
  \"reason\": \"简短原因\",
  \"advice\": \"给 Actor 的修正建议\"
}}

任务：{instruction}
任务类型：{task_type}
任务槽位：{task_slots}
流程标记：{flow_flags}
历史动作：{history_actions[-3:] if history_actions else []}
当前候选动作：{action}
参数：{params}

审核准则：
1) 必须以当前截图可见证据为准，不依赖固定坐标模板。
2) 搜索/输入任务优先保持 TYPE 链路完整：激活输入框 -> TYPE -> 确认。
3) 地图类起终点任务保持双输入顺序：起点后先进入终点输入入口再 TYPE。
4) 视频第N集和评论区任务保持播放优先。
5) 刚 TYPE 后优先确认，不建议再次 TYPE。
6) 仅在目标状态明确达成时允许 COMPLETE。
"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": agent._encode_image(input_data.current_image)}},
        ],
    }]

    try:
        response = agent._call_api(messages, temperature=0.0)
        raw = response.choices[0].message.content
        data = _extract_json_block(raw)
        verdict = str(data.get("verdict", "PASS")).upper()
        if verdict == "REJECT":
            reason = str(data.get("reason", "动作违反规则")).strip()
            advice = str(data.get("advice", "请先处理拦截项，再执行动作")).strip()
            return _review_reject(mailbox, retry_count, f"REJECT: {reason}；建议：{advice}")
    except Exception as e:
        logger.warning(f"Reviewer fallback to PASS: {e}")

    write_packet(
        mailbox,
        channel=A2AChannels.REVIEW,
        sender="reviewer",
        receiver="actor",
        kind="review",
        payload={"reviewer_feedback": "PASS", "retry_count": retry_count, "verdict": "PASS"},
    )
    return {"mailbox": mailbox, "reviewer_feedback": "PASS", "retry_count": retry_count}
