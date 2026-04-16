from typing import Any, Dict, List, Optional, TypedDict

from agent_base import AgentInput, AgentOutput, UsageInfo
from utils.a2a_protocol import A2AMessage


class WorkflowState(TypedDict, total=False):
    input_data: AgentInput
    plan_instruction: str
    task_plan: List[Dict[str, Any]]
    history_actions: List[Dict[str, Any]]
    mailbox: Dict[str, A2AMessage]
    encoded_image_url: str
    current_image_url: str

    # Task profiling for rule-based stage guards.
    task_type: str
    task_slots: Dict[str, str]
    flow_flags: Dict[str, Any]

    proposed_action: str
    proposed_params: Dict[str, Any]
    model_effect: str

    reviewer_feedback: str
    retry_count: int
    review_source: str

    raw_output: str
    usage: Optional[UsageInfo]

    normalized_action: str
    normalized_params: Dict[str, Any]
    expected_effect: str

    final_output: AgentOutput
