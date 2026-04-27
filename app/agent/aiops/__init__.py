"""AIOps 智能运维 Agent 模块

通用 Plan-Execute-Replan 框架实现。
基于 LangGraph 官方教程设计，用于构建能够自主规划和执行的智能代理。

核心设计模式：Plan-Execute-Replan
- Plan（规划）: 将复杂任务分解为可执行的步骤序列
- Execute（执行）: 按顺序执行计划中的每个步骤
- Replan（重规划）: 评估执行结果，决定是继续、调整还是完成

适用场景：
- 复杂的运维诊断任务（需要调用多个工具）
- 需要根据中间结果动态调整策略的场景
- 需要汇总多源信息生成报告的任务

子模块导出：
- state.py: 状态类型定义（PlanExecuteState）
- planner.py: 规划器节点（制定执行计划）
- executor.py: 执行器节点（执行单个步骤）
- replanner.py: 重规划器节点（评估进度，决定下一步）
- utils.py: 通用工具函数
"""

# 从各子模块导入核心组件
from .state import PlanExecuteState        # 状态类型定义
from .planner import planner              # 规划器节点函数
from .executor import executor            # 执行器节点函数
from .replanner import replanner          # 重规划器节点函数

# 定义模块公共 API（用于 from app.agent.aiops import *）
__all__ = [
    "PlanExecuteState",  # 状态类型
    "planner",           # 规划器函数
    "executor",          # 执行器函数
    "replanner",         # 重规划器函数
]
