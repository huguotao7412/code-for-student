import json
import logging
import re
from typing import Any, Dict, List

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)

VALID_ACTIONS = {"CLICK", "SCROLL", "TYPE", "OPEN", "COMPLETE", "CLICK_ID", "ENTER"}


def _reject(retry_count: int, code: str, msg: str) -> Dict[str, Any]:
    return {
        "reviewer_decision": "REJECT",
        "reviewer_feedback": f"REJECT[{code}]: {msg}",
        "violation_code": code,
        "retry_count": retry_count + 1,
        "candidate_completed_task_ids": [],
    }


def _extract_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _build_reviewer_prompt(
    instruction: str,
    task_plan: List[Dict[str, Any]],
    completed_tasks: List[int],
    active_task_id: Any,
    action: str,
    params: Dict[str, Any],
    retry_count: int,
    previous_feedback: str,
) -> str:
    plan_text = json.dumps(task_plan, ensure_ascii=False)
    return f"""你是 GUI 动作审核员。你只审核当前截图上的这一步动作是否合理，不要规划多步。

任务目标：{instruction}
当前候选动作：{action}
当前候选参数：{json.dumps(params, ensure_ascii=False)}
当前优先子任务ID：{active_task_id}
已完成任务ID：{completed_tasks}
总任务计划：{plan_text}
历史重试次数：{retry_count}
上次审核反馈：{previous_feedback or '无'}

审核原则：
审核原则：
1) [最高优排雷]：在审核任何动作前，首先全局扫描屏幕中心和底部区域。发现“稍后更新”、“跳过(Skip)”、“同意并继续”等悬浮弹窗时，必须立刻 REJECT 并要求 Actor 关闭弹窗！
2) [防诱导红线]：如果当前阶段的任务是“输入/TYPE”某个词，但候选动作是 CLICK 且试图去点下方历史记录或推荐列表里的词，必须 REJECT！反馈告诉 Actor：“禁止点击历史记录，请直接在激活的输入框执行 TYPE 动作”。
3) [隐私完结拦截]：如果当前截图明显是【支付、结账页面】或【打车呼叫确认界面】，而候选动作不是 COMPLETE，必须 REJECT 并提示 Actor：“已涉及隐私或支付操作，必须直接输出 COMPLETE:[]”。
4) 只基于当前截图证据判断当前一步是否合理。
5) 参数必须与动作匹配（如 CLICK/SCROLL/TYPE/OPEN 参数结构正确）。
6) 只输出 JSON，不要输出解释。

输出格式（严格 JSON）：
{{
  "decision": "PASS 或 REJECT",
  "feedback": "简短审核意见",
  "violation_code": "违规码或空字符串",
  "candidate_completed_task_ids": [1,2]
}}
"""


def reviewer_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    action = str(state.get("proposed_action", "")).upper().strip()
    params = state.get("proposed_params", {}) or {}
    retry_count = int(state.get("retry_count", 0))

    input_data = state["input_data"]
    task_plan = state.get("task_plan") or []
    completed_tasks = state.get("completed_tasks") or []
    active_task_id = state.get("active_task_id")
    previous_feedback = str(state.get("reviewer_feedback", ""))

    if not action:
        return _reject(retry_count, "EMPTY_ACTION", "未输出动作，请根据当前截图重新决策。")

    if action not in VALID_ACTIONS:
        return _reject(retry_count, "INVALID_ACTION", f"动作 {action} 不在允许集合内。")

    if action in {"CLICK", "SCROLL", "TYPE", "OPEN", "CLICK_ID"} and not isinstance(params, dict):
        return _reject(retry_count, "INVALID_PARAMS", "动作参数不是字典结构。")

    prompt = _build_reviewer_prompt(
        instruction=input_data.instruction or "",
        task_plan=task_plan,
        completed_tasks=completed_tasks,
        active_task_id=active_task_id,
        action=action,
        params=params,
        retry_count=retry_count,
        previous_feedback=previous_feedback,
    )

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
        data = _extract_json(raw)

        decision = str(data.get("decision", "REJECT")).upper().strip()
        feedback = str(data.get("feedback", "")).strip() or "审核模型未给出反馈"
        violation_code = str(data.get("violation_code", "")).strip()

        if decision not in {"PASS", "REJECT"}:
            decision = "REJECT"
            violation_code = violation_code or "INVALID_DECISION"
            feedback = f"{feedback}（decision 非法，已降级 REJECT）"

        candidate_completed_task_ids: List[int] = []
        raw_ids = data.get("candidate_completed_task_ids", [])
        if isinstance(raw_ids, list):
            valid_ids = {int(x.get("id")) for x in task_plan if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
            for item in raw_ids:
                try:
                    tid = int(item)
                except Exception:
                    continue
                if tid in valid_ids and tid not in completed_tasks and tid not in candidate_completed_task_ids:
                    candidate_completed_task_ids.append(tid)

        if decision == "REJECT":
            return {
                "reviewer_decision": "REJECT",
                "reviewer_feedback": f"REJECT[{violation_code or 'MODEL_REJECT'}]: {feedback}",
                "violation_code": violation_code or "MODEL_REJECT",
                "retry_count": retry_count + 1,
                "candidate_completed_task_ids": [],
            }

        return {
            "reviewer_decision": "PASS",
            "reviewer_feedback": "PASS",
            "violation_code": "",
            "candidate_completed_task_ids": candidate_completed_task_ids,
        }
    except Exception as e:
        logger.warning(f"Reviewer VLM failed: {e}")
        return _reject(retry_count, "REVIEWER_VLM_FAIL", f"审核调用失败: {e}")
