"""AIOps 智能运维服务模块

基于 Plan-Execute-Replan 模式的通用任务执行服务。
AIOps = Artificial Intelligence for IT Operations（智能运维）

本服务将 LangGraph 的工作流与 AIOps 诊断场景结合：
- Planner: 分析告警，制定诊断计划
- Executor: 执行诊断步骤（查询日志、监控数据）
- Replanner: 评估进度，决定是否继续或生成报告

架构：LangGraph StateGraph
┌─────────────┐
│   planner   │ ← 入口点，制定诊断计划
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   executor  │ ← 执行计划中的步骤
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  replanner  │ ← 评估进度，决定下一步
└──────┬──────┘
       │
       ├── continue ──→ executor（继续执行）
       │
       └── END ◄──────（生成最终响应）
"""

from typing import AsyncGenerator, Dict, Any  # 异步生成器用于流式输出
from langgraph.graph import StateGraph, END  # LangGraph 状态图和结束节点
from langgraph.checkpoint.memory import MemorySaver  # 内存检查点
from loguru import logger  # 日志记录器

from app.agent.aiops import PlanExecuteState, planner, executor, replanner  # AIOps 节点


# ========== 节点名称常量 ==========
# LangGraph 中节点的唯一标识符
# 在添加节点和定义边时使用这些常量，避免拼写错误
NODE_PLANNER = "planner"     # 规划器节点
NODE_EXECUTOR = "executor"   # 执行器节点
NODE_REPLANNER = "replanner" # 重规划器节点


class AIOpsService:
    """AIOps 智能运维服务

    提供基于 Plan-Execute-Replan 的任务执行能力。

    核心方法：
    - execute(): 执行通用任务（用户自定义任务描述）
    - diagnose(): 执行 AIOps 诊断（固定的任务模板）

    流式输出：
    - 使用 AsyncGenerator 逐步返回执行状态
    - 前端可以实时显示诊断进度
    """

    def __init__(self):
        """初始化 AIOps 服务"""
        # 内存检查点（用于会话持久化，与 RagAgentService 相同）
        self.checkpointer = MemorySaver()

        # 构建 LangGraph 工作流
        self.graph = self._build_graph()

        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 LangGraph 工作流

        工作流定义：
        1. 入口点 → planner（制定计划）
        2. planner → executor（执行步骤）
        3. executor → replanner（评估进度）
        4. replanner 条件边：
           - 有响应 → END（结束）
           - 有剩余计划 → executor（继续执行）
           - 无剩余计划 → END（结束）

        Returns:
            CompiledStateGraph: 编译后的工作流图
        """
        logger.info("构建工作流图...")

        # 创建状态图，指定状态类型
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)        # 规划器：制定计划
        workflow.add_node(NODE_EXECUTOR, executor)      # 执行器：执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)    # 重规划器：评估决策

        # 设置入口点（工作流的起点）
        workflow.set_entry_point(NODE_PLANNER)

        # 定义固定边（无条件的转换）
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)      # planner → executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)    # executor → replanner

        # 定义条件边（根据状态决定转换目标）
        def should_continue(state: PlanExecuteState) -> str:
            """判断工作流是否继续执行

            这个函数会在 replanner 节点之后被调用，
            根据当前状态决定下一步：

            1. 如果已生成最终响应 → END（结束）
            2. 如果还有剩余计划步骤 → executor（继续执行）
            3. 如果没有剩余计划且无响应 → END（生成响应）

            Args:
                state: PlanExecuteState，当前工作流状态

            Returns:
                str: 下一跳目标（END 或 executor）
            """
            # 如果已经生成了最终响应，结束流程
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            # 如果还有计划步骤，继续执行
            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            # 计划为空但没有响应，结束流程（replanner 会生成响应）
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,           # 条件边的源节点
            should_continue,          # 条件判断函数
            {                         # 条件 → 目标节点的映射
                NODE_EXECUTOR: NODE_EXECUTOR,  # 继续执行
                END: END                       # 结束
            }
        )

        # 编译工作流图
        # checkpointer 用于持久化会话状态
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """执行 Plan-Execute-Replan 流程

        通用任务执行接口，接收用户的任务描述，执行完整的工作流。

        Args:
            user_input: 用户的任务描述
                       示例："帮我分析 CPU 使用率过高的问题"
            session_id: 会话 ID（用于会话持久化）

        Yields:
            Dict[str, Any]: 流式事件
                - type: "plan" | "step_complete" | "report" | "complete" | "error"
                - stage: 当前阶段标识
                - message: 描述信息
                - plan/steps/report: 各阶段的具体数据
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,      # 用户任务
                "plan": [],              # 空计划（由 planner 填充）
                "past_steps": [],        # 空历史（由 executor 追加）
                "response": ""           # 空响应（由 replanner 生成）
            }

            # 配置会话
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 流式执行工作流
            # astream() 异步流式执行，每次节点完成时 yield 输出
            # stream_mode="updates" 表示按节点输出
            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 根据节点类型格式化事件
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)
                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)
                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""

            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """AIOps 诊断接口（兼容旧接口）

        执行预定义的 AIOps 诊断任务。
        诊断内容：检查系统告警，分析原因，生成诊断报告。

        诊断报告格式：
        # 告警分析报告
        ## 活跃告警清单
        ## 告警根因分析
        ## 处理方案
        ## 结论

        Args:
            session_id: 会话 ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        from textwrap import dedent

        # 预定义的 AIOps 任务描述
        # 包含详细的任务指令和输出格式要求
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 告警根因分析

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 处理方案

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        # 使用通用 execute 方法执行诊断任务
        async for event in self.execute(aiops_task, session_id):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                # 将 response 包装为 diagnosis 格式
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", "")
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        """格式化 Planner 节点事件

        Args:
            state: planner 节点输出的状态

        Returns:
            Dict: 格式化后的事件字典
        """
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        """格式化 Executor 节点事件

        Args:
            state: executor 节点输出的状态

        Returns:
            Dict: 格式化后的事件字典
        """
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        """格式化 Replanner 节点事件

        Args:
            state: replanner 节点输出的状态

        Returns:
            Dict: 格式化后的事件字典
        """
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估节点运行中"
            }

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            # 已生成最终响应
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            # 重新规划
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan)
            }


# ========== 全局 AIOps 服务单例 ==========
aiops_service = AIOpsService()
