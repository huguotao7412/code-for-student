from typing import Any, Dict, List, Optional, TypedDict

from agent_base import AgentInput, AgentOutput, UsageInfo


class WorkflowState(TypedDict, total=False):
    input_data: AgentInput
    plan_instruction: str
    task_plan: List[Dict[str, Any]]
    history_actions: List[Dict[str, Any]]

    proposed_action: str
    proposed_params: Dict[str, Any]
    model_effect: str

    reviewer_feedback: str
    retry_count: int

    raw_output: str
    usage: Optional[UsageInfo]

    normalized_action: str
    normalized_params: Dict[str, Any]
    expected_effect: str

    final_output: AgentOutput

