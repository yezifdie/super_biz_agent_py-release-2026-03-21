"""Plan-Execute-Replan 状态定义模块

定义 Agent 工作流中流转的数据结构。
LangGraph 使用 TypedDict 来定义状态（State），每个节点函数接收当前状态，返回状态更新。

状态设计原则：
- 最小化：只存储必要的数据，避免状态膨胀
- 可追加：使用 Annotated + operator.add 实现列表的追加式更新
- 类型安全：使用类型注解确保数据正确性
"""

from typing import List, TypedDict, Annotated  # 类型注解
import operator  # 用于定义追加式更新


class PlanExecuteState(TypedDict):
    """Plan-Execute-Replan 工作流的状态定义

    工作流中的每个节点都会接收这个状态对象，并根据需要更新其中的字段。
    LangGraph 会自动合并节点返回的更新到状态中。

    状态字段说明：

    input（输入）
    - 类型：str（字符串）
    - 内容：用户的原始任务描述
    - 来源：工作流开始时由用户输入
    - 示例："分析 CPU 使用率过高的问题"

    plan（执行计划）
    - 类型：List[str]（字符串列表）
    - 内容：计划执行的步骤序列，每一步是一个字符串描述
    - 来源：由 Planner 节点生成
    - 更新：每次执行一个步骤后，该步骤会从列表中移除（plan[1:]）
    - 示例：["查询 CPU 监控数据", "分析错误日志", "生成诊断报告"]

    past_steps（已执行步骤历史）
    - 类型：Annotated[List[tuple], operator.add]（追加式列表）
    - 内容：已执行步骤的元组列表，每个元组为 (步骤描述, 执行结果)
    - 来源：由 Executor 节点追加
    - 特点：使用 operator.add 实现追加更新（而非覆盖），新值会被追加到列表末尾
    - 示例：[("查询 CPU 监控数据", "CPU 使用率 95%"), ("分析错误日志", "发现 OOM 错误")]

    response（最终响应）
    - 类型：str（字符串）
    - 内容：最终返回给用户的报告/回答
    - 来源：由 Replanner 节点在决定 respond 时生成
    - 格式：通常是 Markdown 格式的结构化报告
    """

    # 用户输入（任务描述）
    # 这是整个工作流的起点，描述用户想要完成的任务
    input: str

    # 执行计划（步骤列表）
    # 由 Planner 生成，由 Executor 逐个执行
    # 每执行一步，该步骤就会从列表中移除
    plan: List[str]

    # 已执行的步骤历史
    # 使用 Annotated + operator.add 实现追加式更新
    #
    # 普通列表更新（如 {"past_steps": new_list}）会**替换**整个列表
    # 使用 operator.add 后，更新值会被**追加**到现有列表末尾
    #
    # 示例：
    # 初始状态：past_steps = []
    # Executor 返回：{"past_steps": [("步骤1", "结果1")]}
    # 更新后状态：past_steps = [("步骤1", "结果1")]
    #
    # 再次 Executor 返回：{"past_steps": [("步骤2", "结果2")]}
    # 更新后状态：past_steps = [("步骤1", "结果1"), ("步骤2", "结果2")]
    past_steps: Annotated[List[tuple], operator.add]

    # 最终响应/报告
    # 当 Replanner 决定 "respond" 时，生成最终响应并存储在此字段
    # 之后工作流结束，此响应返回给用户
    response: str
