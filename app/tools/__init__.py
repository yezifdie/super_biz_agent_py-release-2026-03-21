"""工具模块

本模块包含供 Agent（智能代理）调用的各种工具函数。
每个工具都是一个独立的函数，可以被 LLM 调用以获取信息或执行操作。

工具设计原则：
1. 单一职责：每个工具只做一件事
2. 清晰描述：使用 @tool 装饰器提供工具描述
3. 错误处理：工具调用失败时返回有意义的错误信息
4. 工具返回：返回格式化的字符串，便于 LLM 理解和处理

工具列表：
- retrieve_knowledge: 知识检索工具（从向量数据库检索相关文档）
- get_current_time: 时间查询工具（获取当前时间）

LangChain @tool 装饰器：
- 自动将函数转换为 LangChain Tool 对象
- 自动从文档字符串提取工具描述
- 自动从函数签名提取参数信息
- response_format 指定返回格式

MCP 工具 vs 本地工具：
- 本地工具：在 app/tools/ 中定义，直接在 Python 进程中执行
- MCP 工具：通过 MCP 协议调用远程服务（如日志查询、监控数据）
"""

from app.tools.knowledge_tool import retrieve_knowledge  # 知识库检索工具
from app.tools.time_tool import get_current_time        # 时间查询工具

# 定义模块公共 API
__all__ = [
    "retrieve_knowledge",  # 知识检索工具
    "get_current_time",   # 时间查询工具
]
