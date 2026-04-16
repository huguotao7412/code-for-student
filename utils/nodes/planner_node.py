from typing import Any, Dict

from utils.graph_state import WorkflowState


def planner_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]
    need_plan = (
        input_data.step_count <= 1
        or state.get("plan_instruction") != input_data.instruction
        or not state.get("task_plan")
    )
    if need_plan:
        agent._ensure_task_plan(input_data.instruction, input_data.current_image)

    return {
        "plan_instruction": agent._plan_instruction,
        "task_plan": agent._task_plan,
    }

