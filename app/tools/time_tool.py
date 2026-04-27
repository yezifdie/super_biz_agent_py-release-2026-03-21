"""时间查询工具模块

获取当前时间信息的工具。
这是一个简单的工具，用于回答用户关于时间的问题。

使用场景：
- 用户问："现在几点了？"
- 用户问："今天是星期几？"
- Agent 需要知道当前时间用于日志查询范围等操作

技术实现：
- 使用 Python 标准库的 datetime 和 zoneinfo 模块
- 支持时区转换（默认为北京时间 Asia/Shanghai）
- 返回格式化的日期时间字符串

@tool 装饰器：
- 不指定 response_format，使用默认值 "content"
- 返回值直接作为工具调用结果返回给 LLM
"""

from datetime import datetime                    # 日期时间处理
from zoneinfo import ZoneInfo                 # 时区处理（Python 3.9+）

from langchain_core.tools import tool          # LangChain 工具装饰器
from loguru import logger                     # 日志记录器


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间

    当用户询问时间相关问题时，Agent 自动调用此工具。

    调用场景示例：
    - 用户问："现在几点了？"
    - 用户问："今天是星期几？"
    - 用户问："当前日期是什么？"

    技术细节：
    - 使用 Python zoneinfo 模块处理时区（替代已废弃的 pytz）
    - ZoneInfo 是 Python 3.9 引入的标准库
    - 支持所有 IANA 时区标识符

    Args:
        timezone: IANA 时区标识符，默认为 "Asia/Shanghai"（北京时间）
                 其他常用时区：
                 - "UTC": 世界标准时间
                 - "America/New_York": 纽约时间
                 - "Europe/London": 伦敦时间
                 - "Asia/Tokyo": 东京时间

    Returns:
        str: 格式化的日期时间字符串
             格式：YYYY-MM-DD HH:MM:SS
             示例："2026-04-25 14:30:45"

    错误处理：
    - 如果时区标识符无效，返回错误信息字符串
    - 错误信息格式："获取时间失败: [错误原因]"
    """
    try:
        # 获取指定时区的当前时间
        # ZoneInfo 将 IANA 时区标识符转换为时区对象
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)

        # 格式化输出
        # %Y: 4 位年份（如 2026）
        # %m: 2 位月份（01-12）
        # %d: 2 位日期（01-31）
        # %H: 24 小时制小时（00-23）
        # %M: 分钟（00-59）
        # %S: 秒（00-59）
        return now.strftime('%Y-%m-%d %H:%M:%S')

    except Exception as e:
        logger.error(f"时间查询工具调用失败: {e}")
        # 返回错误信息，不抛出异常
        return f"获取时间失败: {str(e)}"
