"""Planner（规划器）节点模块

Planner 是 Plan-Execute-Replan 工作流的第一个节点。
职责：将复杂的用户任务分解为可执行的步骤序列。

工作流程：
1. 查询内部知识库，获取相关经验文档
2. 获取所有可用工具（本地工具 + MCP 工具）
3. 将经验、工具和任务描述组合成提示词
4. 调用 LLM 生成结构化的执行计划

设计决策：
- 使用 qwq-32b 模型（带推理能力）来生成计划，因为规划需要较强的逻辑推理能力
- temperature=0 保证计划输出的确定性
- 先查询知识库，让 LLM 参考历史经验制定更合理的计划
"""

from textwrap import dedent  # 去除多行字符串的缩进
from typing import Dict, Any, List  # 类型注解
from langchain_core.prompts import ChatPromptTemplate  # LangChain 提示词模板
from langchain_qwq import ChatQwen  # 通义千问模型（支持推理）
from pydantic import BaseModel, Field  # Pydantic 模型定义
from loguru import logger  # 日志记录器

from app.config import config  # 全局配置
from app.tools import get_current_time, retrieve_knowledge  # 本地工具
from app.agent.mcp_client import get_mcp_client_with_retry  # MCP 客户端
from .state import PlanExecuteState  # 状态类型定义
from .utils import format_tools_description  # 工具描述格式化


class Plan(BaseModel):
    """计划输出模型

    定义 LLM 生成计划的输出格式。
    使用 Pydantic 确保 LLM 输出符合预期的结构。

    字段说明：
    - steps: 步骤列表，每个步骤是一个字符串描述

    设计考量：
    - 不要求 LLM 输出 JSON，而是通过 prompt 引导其按指定格式输出
    - Pydantic 模型用于结构化输出（配合 .with_structured_output()）
    """
    steps: List[str] = Field(
        description="完成任务所需的不同步骤。这些步骤应该按顺序执行，每一步都建立在前一步的基础上。"
    )


# Planner 提示词模板
# 使用 ChatPromptTemplate 定义对话格式的提示词
planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个专家级别的规划者，你需要将复杂的任务分解为可执行的步骤。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定计划，实际的工具调用由 Executor 负责执行。

                {experience_context}

                对于给定的任务，请创建一个简单的、逐步的计划来完成它。计划应该：
                - 将任务分解为逻辑上独立的步骤
                - 每个步骤应该明确使用哪些工具(如果需要工具的话)来获取信息, 最好能同时提供工具执行所需要的参数
                - 步骤之间应该有清晰的依赖关系
                - 步骤描述要具体、可操作
                - **如果有相关经验文档，请参考其中的方法和步骤制定计划**

                示例输入："分析当前系统的性能问题"
                示例输出（假设有对应工具）：
                步骤1: 使用 get_metrics 工具收集系统的 CPU 和内存使用情况
                步骤2: 使用 query_logs 工具检查最近的错误日志
                步骤3: 使用 query_database 工具分析慢查询日志
                步骤4: 综合以上信息生成性能分析报告
            """).strip(),
        ),
        # 占位符，会在实际调用时被替换为消息列表
        ("placeholder", "{messages}"),
    ]
)


async def planner(state: PlanExecuteState) -> Dict[str, Any]:
    """规划器节点函数

    这是 LangGraph 的节点函数，接收当前状态，返回状态更新。

    执行流程：
    1. 从状态中获取用户输入（input）
    2. 查询内部知识库，获取相关经验文档
    3. 获取所有可用工具的描述
    4. 构建提示词，调用 LLM 生成计划
    5. 返回 {"plan": steps} 更新状态

    Args:
        state: PlanExecuteState，当前工作流状态

    Returns:
        Dict[str, Any]: 状态更新，包含新生成的 plan 字段

    异常处理：
    - 知识库查询失败：记录警告，继续使用空经验上下文
    - LLM 调用失败：返回默认计划（"收集相关信息"等）
    """
    logger.info("=== Planner：制定执行计划 ===")

    input_text = state.get("input", "")
    logger.info(f"用户输入: {input_text}")

    try:
        # 步骤 1：查询内部文档获取相关经验
        # 经验文档可以帮助 LLM 参考历史最佳实践，制定更合理的计划
        logger.info("查询内部文档，寻找相关经验...")
        experience_docs = ""
        try:
            # retrieve_knowledge.ainvoke() 调用知识库检索
            # 返回相关文档的文本内容
            context_str = await retrieve_knowledge.ainvoke({"query": input_text})
            if context_str and context_str.strip():
                experience_docs = context_str
                logger.info(f"找到相关经验文档，长度: {len(experience_docs)}")
            else:
                logger.info("未找到相关经验文档")
        except Exception as e:
            logger.warning(f"查询内部文档失败: {e}")

        # 步骤 2：获取可用工具列表
        # 工具分为两类：
        # - 本地工具：get_current_time、retrieve_knowledge（项目内部实现）
        # - MCP 工具：通过 MCP 协议调用的外部服务工具

        # 获取本地工具（Python 函数）
        local_tools = [
            get_current_time,    # 获取当前时间
            retrieve_knowledge   # 知识库检索
        ]

        # 获取 MCP 工具（从 MCP 服务器发现）
        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        # 合并所有工具
        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        # 格式化工具描述，供 LLM 理解可用工具
        tools_description = format_tools_description(all_tools)

        # 步骤 3：格式化经验文档上下文
        if experience_docs:
            # 将经验文档插入提示词
            experience_context = dedent(f"""
                ## 相关经验文档

                以下是从知识库中检索到的相关经验和最佳实践，请参考这些经验制定执行计划：

                {experience_docs}

                ---
            """).strip()
        else:
            experience_context = ""

        # 步骤 4：创建 LLM 并生成计划
        # 使用 qwq-32b 模型（支持推理），temperature=0 保证确定性输出
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )

        # 构建结构化输出链：提示词 → LLM → Plan 对象
        # with_structured_output(Plan) 确保 LLM 输出符合 Plan 格式
        planner_chain = planner_prompt | llm.with_structured_output(Plan)

        # 调用 LLM 生成计划
        # 传入：消息列表（用户输入）、工具描述、经验上下文
        plan_result = await planner_chain.ainvoke({
            "messages": [("user", input_text)],
            "tools_description": tools_description,
            "experience_context": experience_context
        })

        # 提取步骤列表
        if isinstance(plan_result, Plan):
            plan_steps = plan_result.steps
        else:
            # 兼容处理：某些情况下 LLM 可能返回字典
            plan_steps = plan_result.get("steps", [])

        logger.info(f"计划已生成，共 {len(plan_steps)} 个步骤")
        for i, step in enumerate(plan_steps, 1):
            logger.info(f"  步骤{i}: {step}")

        # 返回状态更新：设置 plan 字段
        return {"plan": plan_steps}

    except Exception as e:
        logger.error(f"生成计划失败: {e}", exc_info=True)
        # 返回默认计划作为后备
        return {
            "plan": [
                "收集相关信息",
                "分析数据",
                "生成报告"
            ]
        }
