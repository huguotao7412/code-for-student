import json
import logging
from typing import Any, Dict

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


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action = str(state.get("proposed_action", "")).upper()
    params = state.get("proposed_params", {}) or {}
    input_data = state["input_data"]
    history_actions = state.get("history_actions") or input_data.history_actions or []
    retry_count = int(state.get("retry_count", 0))

    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()

    instruction = input_data.instruction or ""
    is_search_task = any(k in instruction for k in ["搜索", "查找", "检索", "搜"])
    is_play_task = any(k in instruction for k in ["播放", "收听", "听", "观看"])
    has_typed = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )

    # Rule 0: prevent premature completion before key search/play transitions.
    if action == "COMPLETE" and is_search_task and not has_typed:
        return {
            "reviewer_feedback": "REJECT: 搜索任务尚未完成关键词输入与搜索确认，不能提前 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    # Rule 1: avoid repeated TYPE in consecutive steps.
    if last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 连续执行 TYPE。请优先 ENTER 或点击搜索/确认按钮。",
            "retry_count": retry_count + 1,
        }

    # Rule 2: map flow usually needs entering destination field before another TYPE.
    if "地图" in instruction and last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 地图双输入约束触发。请先点击终点输入入口，再输入终点。",
            "retry_count": retry_count + 1,
        }

    prompt = f"""你是移动端 GUI 动作审核员。请审核 Actor 动作是否违反硬约束。
仅输出 JSON：
{{
  \"verdict\": \"PASS\" 或 \"REJECT\",
  \"reason\": \"简短原因\",
  \"advice\": \"给 Actor 的修正建议\"
}}

任务：{instruction}
历史动作：{history_actions[-3:] if history_actions else []}
当前候选动作：{action}
参数：{params}

审核清单：
1) 若画面有弹窗/广告/权限遮挡，优先处理遮挡。
2) 若动作是 TYPE，需确认输入框已激活（可见 caret）。
3) 地图类任务中，起点后应先进入终点输入入口，再 TYPE 终点。
4) 刚 TYPE 后通常应 ENTER 或点击确认控件，不应盲目再次 TYPE。
5) 搜索任务中，若尚未 TYPE 任务词，不应跳过输入直接点内容结果区。
6) 搜索任务中，完成链路应为“搜索框 TYPE -> 搜索确认(ENTER/搜索按钮) -> 结果选择”。
7) 播放任务中，优先点击播放控件区域；若候选点击更像标题文本区域而非播放控件，应 REJECT 并给修正建议。

补充上下文：
- is_search_task={is_search_task}
- is_play_task={is_play_task}
- has_typed={has_typed}
- last_action={last_action}
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
            return {
                "reviewer_feedback": f"REJECT: {reason}；建议：{advice}",
                "retry_count": retry_count + 1,
            }
    except Exception as e:
        logger.warning(f"Reviewer fallback to PASS: {e}")

    return {"reviewer_feedback": "PASS"}
