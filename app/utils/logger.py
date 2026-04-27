"""日志配置模块

使用 Loguru 配置应用的统一日志系统。
Loguru 是 Python 中比标准 logging 更现代化的日志库，
特点：开箱即用（无需手动配置 Handler/Formatter）、支持彩色输出、自动追踪异常堆栈。

使用方式：
    from app.utils import logger
    logger.info("这是一条信息日志")
    logger.error("错误信息", exc_info=True)
"""

# ========== 标准库 ==========
import sys  # sys.stdout 用于配置控制台输出

# ========== 第三方库 ==========
from loguru import logger  # Loguru 核心，提供全局 logger 实例

# ========== 项目内部依赖 ==========
from app.config import config  # 读取 debug 配置，决定日志详细程度


def setup_logger():
    """配置 Loguru 全局日志系统

    按照 Loguru 最佳实践进行配置，包含两个输出目标：
    1. 控制台输出（带颜色，方便开发时阅读）
    2. 文件输出（按天轮转、自动压缩，用于生产环境排查问题）

    设计考量：
    - 控制台和文件使用不同的格式（文件格式不含 ANSI 颜色码）
    - 文件日志使用异步写入（enqueue=True），避免 IO 阻塞影响请求响应时间
    - 每天生成新日志文件（rotation），保留最近 7 天（retention）
    - 过期日志自动压缩为 zip 格式（compression），节省磁盘空间
    """

    # 步骤 1：移除 Loguru 的默认处理器
    # Loguru 默认会在 stderr 输出，移除后我们可以添加自定义配置
    logger.remove()

    # 步骤 2：添加控制台输出处理器
    # 配置项说明：
    # - sink: 输出目标，sys.stdout 为标准输出（彩色终端会渲染 ANSI 颜色）
    # - format: 日志格式模板，使用 <tag> 语法设置颜色和样式
    #   格式：时间 | 日志级别 | 模块名.函数名:行号 | 消息内容
    # - level: 日志级别，开发模式输出 DEBUG，生产模式输出 INFO
    # - colorize: 启用 ANSI 颜色码（Windows 10+、Linux、macOS 均支持）
    # - backtrace: 启用回溯（显示异常调用链而非单行错误），便于调试
    # - diagnose: DEBUG 模式下显示变量值（敏感信息勿在此模式输出）
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level="DEBUG" if config.debug else "INFO",
        colorize=True,
        backtrace=True,                # 显示完整的异常调用链
        diagnose=config.debug,         # DEBUG 模式下打印变量值（谨慎用于生产）
    )

    # 步骤 3：添加文件输出处理器
    # 配置项说明：
    # - sink: 文件路径模板，{time:YYYY-MM-DD} 会自动替换为当前日期
    #   例如：logs/app_2026-04-25.log
    # - rotation: 每天 0 点自动创建新日志文件（"00:00"）
    # - retention: 7 天后自动删除旧日志（仅保留最近一周的日志）
    # - compression: 过期日志压缩为 zip 格式，大幅节省空间
    # - encoding: UTF-8 编码，确保中文日志不乱码
    # - enqueue: True 表示异步写入（通过独立线程），避免文件 IO 阻塞主线程
    # - level: 文件日志固定为 INFO，不受 debug 模式影响（避免 DEBUG 日志撑爆磁盘）
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="00:00",             # 每天 0 点切割新文件
        retention="7 days",           # 保留最近 7 天的日志
        compression="zip",            # 过期日志压缩为 zip
        encoding="utf-8",             # UTF-8 编码
        enqueue=True,                 # 异步写入，通过队列和独立线程处理
        backtrace=True,              # 显示完整异常堆栈
        diagnose=True,                # 显示变量值，便于排查问题
        level="INFO",                 # 文件日志级别固定为 INFO
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} | {message}",
    )


# ========== 模块加载时立即执行日志配置 ==========
# Python 模块首次被 import 时，这行代码会立即执行
# 这确保了在任何模块使用 logger 之前，日志系统已经正确配置
setup_logger()
