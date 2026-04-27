"""Agent 模块

本模块包含 Agent（智能代理）的核心实现。

Agent 是一种能够自主规划、决策和执行任务的 AI 系统。
与简单的 LLM 调用不同，Agent 可以：
1. 理解复杂的多步骤任务
2. 调用外部工具获取实时信息
3. 根据执行结果动态调整计划
4. 维护任务执行状态和历史

子模块：
- mcp_client.py: MCP（Model Context Protocol）客户端管理
- aiops/: AIOps 智能运维 Agent 的实现
    - state.py: Plan-Execute-Replan 状态定义
    - planner.py: 规划器节点 - 制定执行计划
    - executor.py: 执行器节点 - 执行单个计划步骤
    - replanner.py: 重规划器节点 - 评估进度，决定下一步行动
    - utils.py: 通用工具函数

架构设计（Plan-Execute-Replan 模式）：
┌─────────────┐
│   Planner   │ ← 制定执行计划
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Executor  │ ← 执行计划中的步骤
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Replanner  │ ← 评估进度，决定：继续/调整/响应
└──────┬──────┘
       │
       ├── continue ──→ Executor（继续执行）
       │
       ├── replan ────→ Planner（调整计划）
       │
       └── respond ────→ 结束（生成最终响应）
"""
