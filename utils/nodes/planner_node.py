import json
import logging
from typing import Any, Dict

from utils.graph_state import WorkflowState

logger = logging.getLogger(__name__)


def planner_node(state: WorkflowState, agent: Any) -> Dict[str, Any]:
    input_data = state["input_data"]

    # 判断是否需要重新进行全局规划
    need_plan = (
            input_data.step_count <= 1
            or state.get("plan_instruction") != input_data.instruction
            or not state.get("task_plan")
    )

    if need_plan:
        logger.info(f"[Planner] 启动 A2A 全局规划，分析指令: {input_data.instruction}")

        # 1. 确保生成全局的子任务拆解和 App 提取
        agent._ensure_task_plan(input_data.instruction, input_data.current_image)

        # 2. 为提升任务速度，创建独立的 Actor 代理池以支持并发的任务执行
        # 取代原本的串行流，注入到状态中供后续并发调度
        concurrent_actors = [
            {"actor_id": "actor_explorer", "role": "导航与全局探索", "status": "pending"},
            {"actor_id": "actor_matcher", "role": "目标精准匹配", "status": "pending"}
        ]

        logger.info(f"[Planner] 规划完成，建立并发执行池: {[a['actor_id'] for a in concurrent_actors]}")

        return {
            "plan_instruction": agent._plan_instruction,
            "app_name": agent._app_name,
            "task_plan": agent._task_plan,
            "concurrent_actors": concurrent_actors
        }

    return {}