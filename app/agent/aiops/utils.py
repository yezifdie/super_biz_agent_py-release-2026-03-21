"""AIOps Agent 通用工具函数模块

提供 Agent 内部使用的辅助函数。
"""


from typing import List  # 类型注解


def format_tools_description(tools: List) -> str:
    """将工具列表格式化为描述文本

    将工具对象列表转换为可读的文本描述，供 LLM 理解可用工具。
    每个工具的描述格式为："- 工具名: 工具描述"

    这是为了让 LLM（特别是规划器）能够了解系统提供了哪些工具，
    以便在制定计划时选择合适的工具。

    Args:
        tools: 工具对象列表，每个工具应该有 name 和 description 属性

    Returns:
        str: 格式化后的工具描述文本
             每行格式："- 工具名: 工具描述"
             多行用换行符连接

    示例输出：
        - get_current_time: 获取当前时间
        - search_log: 搜索云日志服务中的日志
        - query_cpu_metrics: 查询 CPU 使用率指标
    """
    tool_descriptions = []
    for tool in tools:
        # 检查工具是否有 name 和 description 属性
        # 这是 LangChain 工具的标准接口
        if hasattr(tool, 'name') and hasattr(tool, 'description'):
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
    return "\n".join(tool_descriptions)
