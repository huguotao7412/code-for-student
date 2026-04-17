from typing import Any, Dict, List

from agent_base import AgentOutput
from utils.graph_state import WorkflowState


MILESTONE_ACTIONS = {"OPEN", "TYPE", "COMPLETE", "CLICK"}


def _merge_completed(current: List[int], candidates: List[int]) -> List[int]:
    seen = set(current)
    merged = list(current)
    for tid in candidates:
        if tid not in seen:
            merged.append(tid)
            seen.add(tid)
    return merged


def format_output_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    som_map = state.get("som_map") or {}
    action, params, expected_effect = agent._normalize_output(
        state.get("proposed_action", "CLICK"),
        state.get("proposed_params", {"point": [500, 500]}),
        som_map=som_map,
    )

    history_actions = state.get("history_actions") or []
    last_action = ""
    if history_actions and isinstance(history_actions[-1], dict):
        last_action = str(history_actions[-1].get("action", "")).upper()

    if last_action == "TYPE" and action == "TYPE":
        action = "CLICK"
        params = {"point": [900, 80]}
        expected_effect = "确认输入并继续"

    model_effect = state.get("model_effect", "")
    if model_effect and not expected_effect:
        expected_effect = model_effect

    reviewer_decision = state.get("reviewer_decision", "PASS")
    hash_changed = bool(state.get("hash_changed", True))
    step_count = int(state["input_data"].step_count)
    last_update_step = int(state.get("last_completed_update_step", -1))
    candidate_ids = state.get("candidate_completed_task_ids") or []

    completed_tasks = list(state.get("completed_tasks") or [])
    can_commit_task = (
        reviewer_decision == "PASS"
        and action in MILESTONE_ACTIONS
        and hash_changed
        and candidate_ids
        and (step_count - last_update_step >= 1)
    )
    if can_commit_task:
        completed_tasks = _merge_completed(completed_tasks, candidate_ids)
        last_update_step = step_count

    final_output = AgentOutput(
        action=action,
        parameters=params,
        raw_output=state.get("raw_output", ""),
        usage=state.get("usage"),
    )

    return {
        "normalized_action": action,
        "normalized_params": params,
        "expected_effect": expected_effect,
        "completed_tasks": completed_tasks,
        "last_completed_update_step": last_update_step,
        "final_output": final_output,
    }
