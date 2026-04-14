import json
import logging
from typing import Any, Dict, List

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


def _safe_point(params: Dict[str, Any]) -> List[int]:
    point = params.get("point") if isinstance(params, dict) else None
    if isinstance(point, list) and len(point) == 2:
        try:
            return [int(point[0]), int(point[1])]
        except Exception:
            return [500, 500]
    return [500, 500]


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action = str(state.get("proposed_action", "")).upper()
    params = state.get("proposed_params", {}) or {}
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
    is_play_task = any(k in instruction for k in ["播放", "收听", "听", "观看"])
    has_typed = any(
        isinstance(item, dict) and str(item.get("action", "")).upper() == "TYPE"
        for item in history_actions
    )
    typed_count = int(flow_flags.get("typed_count", _count_action(history_actions, "TYPE")))
    click_count = int(flow_flags.get("click_count", _count_action(history_actions, "CLICK")))
    _, point_y = _safe_point(params)
    has_episode_target = bool(flow_flags.get("has_episode_target", False))
    needs_comment_area = bool(flow_flags.get("needs_comment_area", False))
    play_started = bool(flow_flags.get("play_started", False))
    search_reactivate_needed = bool(flow_flags.get("search_reactivate_needed", False))

    # Rule 0: prevent premature completion before key transitions.
    if action == "COMPLETE" and (is_search_task or task_type in {"map", "flight", "video"}) and not has_typed:
        return {
            "reviewer_feedback": "REJECT: 关键 TYPE 阶段尚未完成，不能提前 COMPLETE。",
            "retry_count": retry_count + 1,
        }

    # Rule 1: for play/discussion tasks, ensure play state first.
    if task_type == "video" and needs_comment_area and not play_started and action == "CLICK" and point_y > 320:
        return {
            "reviewer_feedback": "REJECT: 讨论区/评论区任务需先进入播放态，再进入评论或讨论区域。",
            "retry_count": retry_count + 1,
        }

    # Rule 2: video 'play episode N' requires play-first then episode selection.
    if task_type == "video" and has_episode_target and not play_started and action == "CLICK" and point_y >= 560:
        return {
            "reviewer_feedback": "REJECT: 播放第N集顺序错误。应先点击播放键进入播放态，再选择具体集数。",
            "retry_count": retry_count + 1,
        }

    # Rule 3: search reactivation across apps after entering search page.
    if search_reactivate_needed and action == "CLICK" and point_y > 220:
        return {
            "reviewer_feedback": "REJECT: 搜索框可能已切换位置。请先重新定位并二次激活搜索框，再 TYPE 任务词。",
            "retry_count": retry_count + 1,
        }

    # Rule 4: strict TYPE-first for search tasks.
    if is_search_task and not has_typed and action == "CLICK" and click_count >= 2 and point_y > 220:
        return {
            "reviewer_feedback": "REJECT: 搜索任务必须先完成搜索框 TYPE，再点内容区。",
            "retry_count": retry_count + 1,
        }

    # Rule 5: slot-driven typing for map/flight start-destination tasks.
    if task_type in {"map", "flight"} and not has_typed and action == "CLICK" and click_count >= 2 and point_y > 220:
        return {
            "reviewer_feedback": "REJECT: 起点/终点任务应先激活输入框并 TYPE 任务词，不能直接点击候选内容区。",
            "retry_count": retry_count + 1,
        }

    # Rule 6: map flow requires entering destination field before another TYPE.
    if task_type == "map" and last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 地图双输入约束触发。请先点击终点输入入口，再输入终点。",
            "retry_count": retry_count + 1,
        }

    # Rule 7: avoid repeated TYPE in consecutive executed steps.
    if last_action == "TYPE" and action == "TYPE":
        return {
            "reviewer_feedback": "REJECT: 连续执行 TYPE。请优先 ENTER 或点击搜索/确认按钮。",
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
任务类型：{task_type}
任务槽位：{task_slots}
流程标记：{flow_flags}
历史动作：{history_actions[-3:] if history_actions else []}
当前候选动作：{action}
参数：{params}

审核清单：
1) 若画面有弹窗/广告/权限遮挡，优先处理遮挡。
2) 搜索任务：必须先 TYPE 任务词，禁止用点击内容区同名词替代 TYPE。
3) 搜索入口点击后若搜索页变化，需重新定位并二次激活新搜索框再 TYPE。
4) 搜索确认键若有多个，优先点击离搜索框最近的确认键。
5) 地图/航班类起终点任务：先激活输入框并 TYPE，再点候选项。
6) 地图类终点输入：起点后先进入终点输入入口，再 TYPE。
7) 视频“播放第N集”：先进入播放态，再选集。
8) 视频评论区/讨论区任务：先进入播放态，再进入评论/讨论区域。
9) 更换语音包任务：先确认界面可见目标词控件，再点击对应词条。
10) 若同名目标词控件出现多个，优先更靠近屏幕中心的候选。
11) 刚 TYPE 后通常应 ENTER 或点击确认控件，不应盲目再次 TYPE。

补充上下文：
- is_search_task={is_search_task}
- is_play_task={is_play_task}
- has_typed={has_typed}
- typed_count={typed_count}
- click_count={click_count}
- play_started={play_started}
- search_reactivate_needed={search_reactivate_needed}
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
