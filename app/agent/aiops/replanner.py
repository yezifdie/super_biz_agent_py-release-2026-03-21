"""Replanner（重规划器）节点模块

Replanner 是 Plan-Execute-Replan 工作流中的决策节点。
职责：根据已执行步骤的结果，决定下一步行动。

三种决策：
1. **continue（继续）**: 当前计划合理，继续执行下一个步骤
2. **replan（重规划）**: 原计划有问题，调整剩余步骤
3. **respond（响应）**: 信息已充足，生成最终响应

设计原则：
- **信息充足就响应**：不追求完美的计划，"足够好"就应该结束
- **逐步简化**：replan 时新步骤数应 <= 剩余步骤数（简化而非扩展）
- **安全限制**：已执行步骤 >= 5 时，禁止 replan，强制 respond
- **最大步骤限制**：已执行步骤 >= 8 时，强制生成响应

决策优先级：
respond > continue > replan

这是为了避免：
- 无限循环（不断执行步骤而不结束）
- 过度规划（replan 产生更多步骤）
- 响应延迟（等待完美的信息收集）
"""

from textwrap import dedent  # 去除多行字符串的缩进
from typing import Dict, Any, List  # 类型注解
from langchain_core.prompts import ChatPromptTemplate  # LangChain 提示词模板
from langchain_qwq import ChatQwen  # 通义千问模型
from pydantic import BaseModel, Field  # Pydantic 模型定义
from loguru import logger  # 日志记录器

from app.config import config  # 全局配置
from app.tools import get_current_time, retrieve_knowledge  # 本地工具
from app.agent.mcp_client import get_mcp_client_with_retry  # MCP 客户端
from .state import PlanExecuteState  # 状态类型定义
from .utils import format_tools_description  # 工具描述格式化


class Response(BaseModel):
    """最终响应模型

    定义生成最终响应时的输出格式。
    """
    response: str = Field(description="对用户的最终响应")


class Act(BaseModel):
    """重规划决策模型

    定义 Replanner 的输出格式，包含：
    - action: 下一步行动（continue / replan / respond）
    - new_steps: 当 action=replan 时的新步骤列表
    """
    action: str = Field(
        description="""下一步的行动，必须是以下三种之一：
        - 'continue': 当前计划合理，继续执行下一个步骤
        - 'replan': 当前计划需要调整，提供新的步骤列表
        - 'respond': 计划已完成且信息充足，生成最终响应"""
    )
    # action 为 'replan' 时，新的步骤列表（会替换当前剩余计划）
    new_steps: List[str] = Field(
        default_factory=list,
        description="新的步骤列表（如果 action 是 'replan'，这些步骤会替换剩余计划）"
    )


# Replanner 提示词模板
# 核心是指示 LLM 如何在三种行动之间做决策
replanner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                作为一个重新规划专家，你需要根据已执行的步骤决定下一步行动。

                可用工具列表（用于制定计划时参考）：

                {tools_description}

                注意：你的职责是制定或调整计划，实际的工具调用由 Executor 负责执行。

                你有三个选择（按优先级排序）：

                **1. 'respond' - 信息充足，立即生成最终响应** 【最高优先级】
                   - 使用场景：当前信息已经足够回答用户问题
                   - 决策标准：
                     * 已执行步骤 >= 3 且获取了关键信息
                     * 或者已执行步骤 >= 5（无论结果如何）
                     * 或者当前信息完全满足任务需求
                   - ⚠️ 不要等到"完美"才响应，"足够好"就应该立即 respond

                **2. 'continue' - 当前计划合理，继续执行** 【次优先级】
                   - 使用场景：剩余计划合理且必要
                   - 决策标准：剩余步骤确实能提供关键信息
                   - ⚠️ 如果剩余步骤不是"必需"的，应选择 respond

                **3. 'replan' - 当前计划有严重问题** 【最低优先级，谨慎使用】
                   - 使用场景：原计划明显错误或遗漏关键步骤
                   - ⚠️ **严格限制**：
                     * 新步骤数量必须 <= 当前剩余步骤数
                     * 优先简化计划，不要添加不必要的步骤
                     * 总步骤数已执行 >= 5 次时，禁止 replan，只能 respond

                评估标准：
                - 当前信息是否已经足够解决用户问题？【最关键】
                - 已执行步骤是否成功获取了核心信息？
                - 剩余步骤是否真的"必需"？
                - 已执行步骤数是否过多（>= 5）？如果是，立即 respond

                **决策优先级口诀：**
                "优先结束 > 保持不变 > 调整计划"
                "信息足够就响应，不要追求完美"
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)

# 最终响应生成提示词
response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            dedent("""
                根据原始任务和已执行步骤的结果，生成一个全面的最终响应。

                响应要求：
                - 清晰、结构化
                - 基于实际数据，不要编造
                - 如果某些步骤失败，要诚实说明
                - 使用 Markdown 格式
            """).strip(),
        ),
        ("placeholder", "{messages}"),
    ]
)


async def replanner(state: PlanExecuteState) -> Dict[str, Any]:
    """重规划器节点函数

    评估当前进度，决定下一步行动。

    决策流程：
    1. 检查是否超过最大步骤限制（>= 8），是则强制生成响应
    2. 获取可用工具描述
    3. 格式化已执行步骤的摘要
    4. 如果还有剩余计划：
       - 调用 LLM 做决策（continue / replan / respond）
       - 根据决策执行相应操作
    5. 如果没有剩余计划，直接生成响应

    Args:
        state: PlanExecuteState，当前工作流状态

    Returns:
        Dict[str, Any]: 状态更新，可能包含：
        - plan: 新步骤列表（replan 时）
        - response: 最终响应（respond 时）
        - 空字典（continue 时，无状态变化）
    """
    logger.info("=== Replanner：重新规划 ===")

    input_text = state.get("input", "")
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])

    logger.info(f"剩余计划步骤: {len(plan)}")
    logger.info(f"已执行步骤: {len(past_steps)}")

    # ⚠️ 安全限制：如果已执行步骤过多，强制生成最终响应
    # 避免无限循环，确保系统能在合理时间内返回结果
    MAX_STEPS = 8
    if len(past_steps) >= MAX_STEPS:
        logger.warning(f"已执行 {len(past_steps)} 个步骤，超过最大限制 {MAX_STEPS}，强制生成最终响应")
        llm = ChatQwen(
            model=config.rag_model,
            api_key=config.dashscope_api_key,
            temperature=0
        )
        return await _generate_response(state, llm)

    # 获取可用工具列表（用于提示词）
    try:
        local_tools = [
            get_current_time,
            retrieve_knowledge
        ]

        mcp_client = await get_mcp_client_with_retry()
        mcp_tools = await mcp_client.get_tools()

        all_tools = local_tools + mcp_tools
        logger.info(f"可用工具数量: 本地 {len(local_tools)} + MCP {len(mcp_tools)}")

        tools_description = format_tools_description(all_tools)
    except Exception as e:
        logger.warning(f"获取工具列表失败: {e}")
        tools_description = "无法获取工具列表"

    # 创建 LLM 实例
    llm = ChatQwen(
        model=config.rag_model,
        api_key=config.dashscope_api_key,
        temperature=0
    )

    # 格式化已执行步骤的摘要（用于 LLM 评估）
    # 限制每个结果的长度（最多 300 字符），避免上下文过长
    steps_summary = "\n".join([
        f"步骤: {step}\n结果: {result[:300]}..."
        for step, result in past_steps
    ])

    # 如果还有剩余计划，进行决策
    if plan:
        logger.info("还有剩余计划，评估下一步行动")

        # 构建结构化输出链
        replanner_chain = replanner_prompt | llm.with_structured_output(Act)

        try:
            # 构建发给 LLM 的消息
            messages = [
                ("user", f"原始任务: {input_text}"),
                ("user", f"已执行的步骤:\n{steps_summary}"),
                ("user", f"剩余计划: {', '.join(plan)}"),
                ("user", f"⚠️ 重要提示：已执行 {len(past_steps)} 个步骤，请优先考虑是否信息已足够生成响应（respond）")
            ]

            # 调用 LLM 做决策
            act = await replanner_chain.invoke({
                "messages": messages,
                "tools_description": tools_description
            })

            # 处理返回结果
            if isinstance(act, Act):
                action = act.action
                new_steps = act.new_steps
            else:
                # 兼容处理：某些情况下可能返回字典
                action = act.get("action", "continue")
                new_steps = act.get("new_steps", [])

            logger.info(f"Replanner 决策: {action}")

            # 根据决策执行相应操作
            if action == "respond":
                # 信息充足，生成最终响应
                logger.info("决定生成最终响应")
                return await _generate_response(state, llm)

            elif action == "replan":
                # 调整计划
                # ⚠️ 强制限制：新步骤数不能超过当前剩余步骤数
                if len(new_steps) > len(plan):
                    logger.warning(
                        f"新步骤数 {len(new_steps)} > 剩余步骤数 {len(plan)}，"
                        f"强制截断为 {len(plan)} 个步骤"
                    )
                    new_steps = new_steps[:len(plan)]

                # ⚠️ 二次检查：已执行步骤 >= 5 时禁止 replan
                if len(past_steps) >= 5:
                    logger.warning(f"已执行 {len(past_steps)} 个步骤，禁止重新规划，强制生成响应")
                    return await _generate_response(state, llm)

                logger.info(f"决定调整计划，新步骤数量: {len(new_steps)}")
                if new_steps:
                    # 替换剩余计划
                    return {"plan": new_steps}
                else:
                    # replan 但未提供新步骤，继续执行原计划
                    logger.warning("replan 但未提供新步骤，继续执行原计划")
                    return {}

            else:  # action == "continue"
                # 继续执行当前计划（无状态变化）
                logger.info("决定继续执行当前计划")
                return {}

        except Exception as e:
            logger.error(f"重新规划失败: {e}, 继续执行剩余计划")
            return {}

    else:
        # 没有剩余计划，生成最终响应
        logger.info("计划已执行完毕，生成最终响应")
        return await _generate_response(state, llm)


async def _generate_response(state: PlanExecuteState, llm: ChatQwen) -> Dict[str, Any]:
    """生成最终响应（内部函数）

    将执行历史汇总，调用 LLM 生成结构化的最终报告。

    Args:
        state: 当前工作流状态
        llm: 已初始化的 LLM 实例

    Returns:
        Dict[str, Any]: 包含 response 字段的状态更新
    """
    logger.info("生成最终响应...")

    input_text = state.get("input", "")
    past_steps = state.get("past_steps", [])

    # 格式化执行历史（用于 LLM 生成报告）
    execution_history = "\n\n".join([
        f"### 步骤: {step}\n**结果:**\n{result}"
        for step, result in past_steps
    ])

    # 构建响应生成链
    response_gen = response_prompt | llm.with_structured_output(Response)

    try:
        messages = [
            ("user", f"原始任务: {input_text}"),
            ("user", f"执行历史:\n{execution_history}"),
            ("user", "请基于以上信息生成全面的最终响应")
        ]

        response_obj = await response_gen.invoke({"messages": messages})

        # 处理返回结果
        if isinstance(response_obj, Response):
            final_response = response_obj.response
        else:
            final_response = response_obj.get("response", "")

        logger.info(f"最终响应生成完成，长度: {len(final_response)}")

        return {"response": final_response}

    except Exception as e:
        logger.error(f"生成响应失败: {e}")
        # 生成后备响应（简单的步骤汇总）
        fallback_response = f"""# 任务执行结果

## 原始任务
{input_text}

## 执行的步骤
{_format_simple_steps(past_steps)}

## 说明
由于系统异常，无法生成完整响应。以上是已收集的信息。
"""
        return {"response": fallback_response}


def _format_simple_steps(past_steps: list) -> str:
    """格式化步骤列表（简单版，用于后备响应）"""
    if not past_steps:
        return "无"

    formatted = []
    for i, (step, result) in enumerate(past_steps, 1):
        # 截断过长的结果
        result_preview = result[:200] + "..." if len(result) > 200 else result
        formatted.append(f"{i}. **{step}**\n   {result_preview}\n")

    return "\n".join(formatted)
