from typing import Any, Dict

from utils.a2a_protocol import A2AChannels, ensure_mailbox, write_packet
from utils.graph_state import WorkflowState


def planner_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    current_image_url = str(state.get("current_image_url", "") or "")
    need_plan = (
        input_data.step_count <= 1
        or state.get("plan_instruction") != input_data.instruction
        or not state.get("task_plan")
    )
    if need_plan:
        agent._ensure_task_plan(input_data.instruction, input_data.current_image, current_image_url=current_image_url)

    mailbox = ensure_mailbox(state)
    write_packet(
        mailbox,
        channel=A2AChannels.TASK_CONTEXT,
        sender="planner",
        receiver="actor",
        kind="context",
        payload={
            "instruction": input_data.instruction,
            "step_count": input_data.step_count,
            "history_actions": input_data.history_actions,
        },
    )
    write_packet(
        mailbox,
        channel=A2AChannels.PLAN,
        sender="planner",
        receiver="actor",
        kind="plan",
        payload={
            "plan_instruction": agent._plan_instruction,
            "task_plan": agent._task_plan,
        },
    )

    return {
        "mailbox": mailbox,
        "plan_instruction": agent._plan_instruction,
        "task_plan": agent._task_plan,
    }
