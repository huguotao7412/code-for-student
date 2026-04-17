from typing import Any, Dict, List, Optional, TypedDict

from agent_base import AgentInput, AgentOutput, UsageInfo


class WorkflowState(TypedDict, total=False):
    input_data: AgentInput

    plan_instruction: str
    app_name: str
    task_plan: List[Dict[str, Any]]
    history_actions: List[Dict[str, Any]]

    completed_tasks: List[int]
    active_task_id: Optional[int]
    candidate_completed_task_ids: List[int]
    last_completed_update_step: int

    som_map: Dict[int, Dict[str, Any]]
    frame_hash_global: str
    frame_hash_local: str
    hash_changed: bool

    proposed_action: str
    proposed_params: Dict[str, Any]
    model_effect: str

    reviewer_decision: str
    reviewer_feedback: str
    violation_code: str
    retry_count: int

    raw_output: str
    usage: Optional[UsageInfo]

    normalized_action: str
    normalized_params: Dict[str, Any]
    expected_effect: str

    final_output: AgentOutput

