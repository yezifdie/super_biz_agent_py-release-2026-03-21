"""Executor（执行器）节点模块

Executor 是 Plan-Execute-Replan 工作流的核心执行节点。
职责：执行计划中的下一个步骤，并记录执行结果。

工作流程：
1. 从状态中获取当前计划（plan）和第一个待执行步骤（task）
2. 获取所有可用工具
3. 构建包含 SystemMessage 和 HumanMessage 的消息列表
4. 调用 LLM，让其决定是否需要调用工具
5. 如果需要工具调用，使用 ToolNode 自动执行工具
6. 将执行结果记录到 past_steps，并从 plan 中移除已执行步骤
7. 返回状态更新

设计决策：
- 使用 LangGraph 的 ToolNode 处理工具调用，自动管理工具调用的循环
- ToolNode 会自动处理工具结果的返回和再次调用（如果 LLM 认为需要）
- 每次只执行一个步骤，保持状态的可追踪性
"""

from typing import Dict, Any  # 类型注解
from langchain_core.messages import HumanMessage, SystemMessage  # LangChain 消息类型
from langchain_qwq import ChatQwen  # 通义千问模型
from langgraph.prebuilt import ToolNode  # LangGraph 预建的工具节点
from loguru import logger  # 日志记录器

from app.config import config  # 全局配置
from app.tools import get_current_time, retrieve_knowledge  # 本地工具
from app.agent.mcp_client import get_mcp_client_with_retry  # MCP 客户端
from .state import PlanExecuteState  # 状态类型定义


async def executor(state: PlanExecuteState) -> Dict[str, Any]:
    """执行器节点函数

    执行计划中的下一个步骤。

    执行逻辑：
    1. 获取当前步骤（plan[0]）
    2. 准备工具列表和 LLM
    3. 调用 LLM + 工具执行步骤
    4. 记录执行结果到 past_steps
    5. 从 plan 中移除已执行的步骤
    6. 返回状态更新

    Args:
        state: PlanExecuteState，当前工作流状态

    Returns:
        Dict[str, Any]: 状态更新，包含：
        - plan: 移除第一个步骤后的剩余计划
        - past_steps: 追加执行记录（通过 operator.add）

    异常处理：
    - 工具调用失败：记录错误，但继续执行剩余步骤（使用错误信息作为结果）
    """
    logger.info("=== Executor：执行步骤 ===")

    plan = state.get("plan", [])

    # 如果计划为空，不执行任何操作
    if not plan:
        logger.info("计划为空，跳过执行")
        return {}

    # 取出第一个步骤作为当前任务
    task = plan[0]
    logger.info(f"当前任务: {task}")

    try:
        # 步骤 1：获取可用工具列表
        # 与 Planner 相同，需要获取本地工具和 MCP 工具
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        # 获取 MCP 工具
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 合并所有工具
        all_tools = local_tools + mcp_tools

        # 步骤 2：创建 LLM 并绑定工具
        # bind_tools() 让 LLM 能够识别和调用工具
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(all_tools)

        # 步骤 3：创建 ToolNode
        # ToolNode 是 LangGraph 预建的处理工具调用的节点
        # 它会：
        # 1. 接收 LLM 的 tool_calls
        # 2. 调用对应的工具
        # 3. 返回工具结果作为消息
        tool_node = ToolNode(all_tools)

        # 步骤 4：构建消息列表
        # SystemMessage 提供执行上下文的指导
        # HumanMessage 提供具体的任务描述
        messages = [
            SystemMessage(content="""你是一个能力强大的助手，负责执行具体的任务步骤。

你可以使用各种工具来完成任务。对于每个步骤：
1. 理解步骤的目标
2. 选择合适的工具，如果已经指定了工具，则使用指定的工具
3. 调用工具获取信息
4. 返回执行结果

注意：
- 如果工具调用失败，请说明失败原因
- 不要编造数据，只返回实际获取的信息
- 执行结果要清晰、准确
- 专注于当前步骤，不要考虑其他任务"""),
            HumanMessage(content=f"请执行以下任务: {task}")
        ]

        # 步骤 5：LLM 决定是否调用工具
        # LLM 会分析任务，判断是否需要调用工具来获取信息
        llm_response = await llm_with_tools.ainvoke(messages)
        logger.info(f"LLM 响应类型: {type(llm_response)}")

        # 步骤 6：如果 LLM 决定调用工具，执行工具
        if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
            logger.info(f"检测到 {len(llm_response.tool_calls)} 个工具调用")

            # 将 LLM 响应追加到消息列表
            messages.append(llm_response)

            # 使用 ToolNode 执行工具调用
            # ToolNode 会处理工具调用的循环（LLM 可能需要多次调用工具）
            tool_messages = await tool_node.ainvoke({"messages": messages})

            # 步骤 7：将工具结果追加到消息列表，再次调用 LLM 生成最终答案
            messages.extend(tool_messages["messages"])
            final_response = await llm_with_tools.ainvoke(messages)

            # 提取响应内容
            result = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # LLM 决定不需要调用工具，直接使用 LLM 的输出作为结果
            # 这适用于：步骤是纯分析/推理任务，或者任务已通过之前的步骤完成
            logger.info("LLM 未调用工具，直接返回结果")
            result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

        logger.info(f"步骤执行完成，结果长度: {len(result)}")

        # 步骤 8：返回状态更新
        # - plan: 移除第一个步骤（使用切片 plan[1:]）
        # - past_steps: 追加 (task, result) 元组
        #   注意：past_steps 使用 operator.add，所以新值会追加而非替换
        return {
            "plan": plan[1:],           # 移除已执行的第一个步骤
            "past_steps": [(task, result)],  # 追加执行记录
        }

    except Exception as e:
        logger.error(f"执行步骤失败: {e}", exc_info=True)
        # 即使执行失败，也记录到 past_steps，以便 Replanner 知道发生了什么
        return {
            "plan": plan[1:],  # 移除失败的步骤
            "past_steps": [(task, f"执行失败: {str(e)}")],  # 记录失败信息
        }
