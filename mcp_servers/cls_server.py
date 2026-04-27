"""腾讯云 CLS (Cloud Log Service) MCP Server

本地实现的 CLS 日志服务 MCP Server，提供日志查询、检索和分析功能。

MCP Server 架构：
- 使用 FastMCP 框架构建
- 暴露多个工具函数供 Agent 调用
- 运行在独立的进程中（端口 8003）
- 主应用通过 langchain-mcp-adapters 连接

提供的工具：
- get_current_timestamp: 获取当前时间戳（毫秒）
- get_region_code_by_name: 根据地区名称查找地区代码
- get_topic_info_by_name: 根据主题名称查找日志主题信息
- search_topic_by_service_name: 根据服务名称搜索日志主题
- search_log: 在指定主题中搜索日志

注意：当前实现使用 Mock 数据，实际生产环境需要连接真实的腾讯云 CLS API。
"""

import logging          # 标准库日志模块
import functools        # 函数装饰器工具（wraps）
import json            # JSON 序列化
from typing import Dict, Any, Optional  # 类型注解
from datetime import datetime, timedelta  # 日期时间处理
from fastmcp import FastMCP  # FastMCP 框架

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CLS_MCP_Server")

# 创建 FastMCP 实例
# FastMCP 是构建 MCP 服务器的框架，自动处理协议细节
mcp = FastMCP("CLS")


def log_tool_call(func):
    """工具调用日志装饰器

    记录每个工具函数的调用信息，包括：
    - 方法名称
    - 输入参数
    - 返回状态
    - 返回结果摘要

    使用 functools.wraps 保持原函数的元数据（名称、文档等）。

    使用方式：
        @mcp.tool()
        @log_tool_call
        def my_tool(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用分隔线和方法名
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数信息
        if kwargs:
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录成功状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                # 只显示前 5 个字段的摘要
                summary = {
                    k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                    for k, v in list(result.items())[:5]
                }
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间

    统一处理时间参数的解析，支持：
    1. 字符串格式："YYYY-MM-DD HH:MM:SS"
    2. 默认时间 + 偏移量

    Args:
        time_str: 时间字符串，如果为 None 则使用默认时间
        default_offset_hours: 默认时间的偏移小时数
                              正数：未来偏移
                              负数：过去偏移

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int) -> str:
    """生成基于基准时间的时间字符串

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量（正数为未来，负数为过去）

    Returns:
        str: 格式化的时间字符串 "YYYY-MM-DD HH:MM:SS"
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
@log_tool_call
def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）

    提供标准化的毫秒时间戳，用于：
    1. 作为 search_log 的 end_time 参数（查询到现在）
    2. 计算历史时间点作为 start_time 参数

    时间戳转换示例：
    - Python: datetime.now().timestamp() * 1000
    - JavaScript: Date.now()

    Returns:
        int: 当前时间戳（毫秒），例如: 1708012345000

    使用示例:
        # 获取当前时间
        current = get_current_timestamp()

        # 计算15分钟前的时间
        fifteen_min_ago = current - (15 * 60 * 1000)

        # 用于搜索最近15分钟的日志
        search_log(
            topic_id="topic-001",
            start_time=fifteen_min_ago,
            end_time=current
        )
    """
    return int(datetime.now().timestamp() * 1000)


@mcp.tool()
@log_tool_call
def get_region_code_by_name(region_name: str) -> Dict[str, Any]:
    """根据地区名称搜索对应的地区参数

    将人类可读的地区名称转换为腾讯云 API 所需的地区代码。

    Args:
        region_name: 地区名称（如：北京、上海、广州等）

    Returns:
        Dict: 包含地区代码和相关信息的字典
            - region_code: 腾讯云地区代码（如 "ap-beijing"）
            - region_name: 地区名称
            - available: 是否可用
            - error: 如果地区不存在，包含错误信息

    示例:
        get_region_code_by_name("北京")
        # 返回: {"region_code": "ap-beijing", "region_name": "北京", "available": True}
    """
    # 模拟地区映射表
    # 实际生产环境应从腾讯云 API 或配置文件中读取
    region_mapping = {
        "北京": {"region_code": "ap-beijing", "region_name": "北京", "available": True},
        "上海": {"region_code": "ap-shanghai", "region_name": "上海", "available": True},
        "广州": {"region_code": "ap-guangzhou", "region_name": "广州", "available": True},
    }

    result = region_mapping.get(region_name)
    if result:
        return result
    else:
        return {
            "region_code": None,
            "region_name": region_name,
            "available": False,
            "error": f"未找到地区: {region_name}"
        }


@mcp.tool()
@log_tool_call
def get_topic_info_by_name(topic_name: str, region_code: Optional[str] = None) -> Dict[str, Any]:
    """根据主题名称搜索相关的主题信息

    在腾讯云 CLS 中，日志主题（Topic）是存储日志的容器。
    每个服务通常对应一个或多个日志主题。

    Args:
        topic_name: 主题名称
        region_code: 地区代码（可选，用于过滤）

    Returns:
        Dict: 包含主题信息的字典
            - topic_id: 主题ID（用于后续日志查询）
            - topic_name: 主题名称
            - service_name: 服务名称
            - region_code: 所属地区
            - create_time: 创建时间
            - log_count: 日志数量
            - description: 主题描述
    """
    # Mock 主题数据
    mock_topics = [
        {
            "topic_id": "topic-alerts",
            "topic_name": "告警日志",
            "service_name": "alert-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "系统告警日志，包含 CPU、内存、网络等监控告警"
        },
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "服务应用日志"
        }
    ]

    # 根据名称和地区筛选
    for topic in mock_topics:
        if topic["topic_name"] == topic_name:
            if region_code is None or topic["region_code"] == region_code:
                return topic

    return {
        "topic_id": None,
        "topic_name": topic_name,
        "region_code": region_code,
        "error": f"未找到主题: {topic_name}"
    }


@mcp.tool()
@log_tool_call
def search_topic_by_service_name(
    service_name: str,
    region_code: Optional[str] = None,
    fuzzy: bool = True
) -> Dict[str, Any]:
    """根据服务名称搜索相关的日志主题信息

    这是日志查询的关键前置步骤：先找到服务对应的日志主题（topic_id），
    再使用 topic_id 进行实际日志查询。

    搜索模式：
    - 模糊搜索（默认）：部分匹配，如 "sync" 可匹配 "data-sync-service"
    - 精确搜索：必须完全一致

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service", "sync", "data-sync"
        region_code: 地区代码（可选）
            示例: "ap-beijing", "ap-shanghai"
        fuzzy: 是否启用模糊搜索（默认 True）
            True: 部分匹配
            False: 精确匹配

    Returns:
        Dict: 搜索结果
            - total: 匹配到的主题数量
            - topics: 主题列表
            - query: 查询条件
            - message: 结果描述

    使用示例:
        # 模糊搜索（推荐）
        search_topic_by_service_name(service_name="data-sync")

        # 精确搜索
        search_topic_by_service_name(
            service_name="data-sync-service",
            fuzzy=False
        )

        # 完整流程
        result = search_topic_by_service_name(service_name="data-sync-service")
        topic_id = result["topics"][0]["topic_id"]
        # 然后使用 topic_id 调用 search_log
    """
    # Mock 主题数据
    mock_topics = [
        {
            "topic_id": "topic-alerts",
            "topic_name": "告警日志",
            "service_name": "alert-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "系统告警日志，包含 CPU、内存、网络等监控告警"
        },
        {
            "topic_id": "topic-001",
            "topic_name": "数据同步服务日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的应用日志，包含同步任务执行情况"
        },
        {
            "topic_id": "topic-002",
            "topic_name": "数据同步服务错误日志",
            "service_name": "data-sync-service",
            "region_code": "ap-beijing",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "数据同步服务的错误日志"
        },
        {
            "topic_id": "topic-003",
            "topic_name": "API网关服务日志",
            "service_name": "api-gateway-service",
            "region_code": "ap-shanghai",
            "create_time": "2024-01-01 10:00:00",
            "log_count": 0,
            "description": "API网关服务日志"
        }
    ]

    matched_topics = []

    # 搜索逻辑
    for topic in mock_topics:
        # 地区筛选
        if region_code and topic["region_code"] != region_code:
            continue

        topic_service_name = topic.get("service_name", "")

        if fuzzy:
            # 模糊匹配：查询字符串包含在服务名中，或服务名包含在查询字符串中
            if (service_name.lower() in topic_service_name.lower() or
                topic_service_name.lower() in service_name.lower()):
                matched_topics.append(topic)
        else:
            # 精确匹配
            if topic_service_name == service_name:
                matched_topics.append(topic)

    return {
        "total": len(matched_topics),
        "topics": matched_topics,
        "query": {
            "service_name": service_name,
            "region_code": region_code,
            "fuzzy": fuzzy
        },
        "message": f"找到 {len(matched_topics)} 个匹配的日志主题" if matched_topics else f"未找到服务 '{service_name}' 的日志主题"
    }


@mcp.tool()
@log_tool_call
def search_log(
    topic_id: str,
    start_time: int | str,
    end_time: int | str,
    query: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """基于提供的查询参数搜索日志

    在指定的日志主题中搜索符合条件的日志条目。
    支持时间范围过滤和关键词搜索。

    参数说明：
    - topic_id: 日志主题 ID，从 search_topic_by_service_name 获取
    - start_time/end_time: 支持毫秒时间戳或字符串格式
    - query: CLS 查询语法，如 "level:ERROR" 或 "message:异常"
    - limit: 返回结果数量上限

    Args:
        topic_id: 主题ID（必填）
            示例: "topic-001", "topic-alerts"
        start_time: 开始时间（必填）
            - int: 毫秒时间戳
            - str: "YYYY-MM-DD HH:MM:SS"
        end_time: 结束时间（必填）
            - int: 毫秒时间戳
            - str: "YYYY-MM-DD HH:MM:SS"
        query: 查询语句（可选，CLS 查询语法）
            示例: "level:ERROR" 或 "message:异常"
        limit: 返回结果数量限制（默认 100）

    Returns:
        Dict: 搜索结果
            - topic_id: 主题ID
            - start_time: 开始时间戳
            - end_time: 结束时间戳
            - query: 查询语句
            - limit: 结果限制
            - total: 实际返回的日志条数
            - logs: 日志列表
            - took_ms: 查询耗时（毫秒）
            - message: 查询状态消息

    使用示例:
        # 使用字符串时间格式
        search_log(
            topic_id="topic-alerts",
            start_time="2026-04-25 10:00:00",
            end_time="2026-04-25 11:00:00",
            limit=100
        )

        # 使用时间戳格式
        current_ts = get_current_timestamp()
        start_ts = current_ts - (15 * 60 * 1000)
        search_log(
            topic_id="topic-001",
            start_time=start_ts,
            end_time=current_ts,
            limit=100
        )

        # 搜索告警日志
        search_log(
            topic_id="topic-alerts",
            start_time="2026-04-25 10:00:00",
            end_time="2026-04-25 11:00:00",
            query="level:ERROR",
            limit=50
        )
    """
    # 统一解析 start_time 和 end_time 为毫秒时间戳
    if isinstance(start_time, str):
        start_ms = int(parse_time_or_default(start_time, default_offset_hours=-1).timestamp() * 1000)
    else:
        start_ms = start_time

    if isinstance(end_time, str):
        end_ms = int(parse_time_or_default(end_time, default_offset_hours=0).timestamp() * 1000)
    else:
        end_ms = end_time

    # 根据 topic_id 返回不同的结果（Mock 数据）
    if topic_id == "topic-alerts":
        # 告警日志：返回包含告警信息的日志列表
        alert_logs = [
            {
                "timestamp": "2026-04-25 10:15:23",
                "level": "CRITICAL",
                "message": "[告警] CPU 使用率超过 95%，持续超过 5 分钟",
                "alert_id": "ALT-001",
                "alert_type": "cpu_high",
                "severity": "critical",
                "service": "data-sync-service",
                "value": "96.5%"
            },
            {
                "timestamp": "2026-04-25 10:18:45",
                "level": "WARNING",
                "message": "[告警] 内存使用率超过 80%，接近阈值",
                "alert_id": "ALT-002",
                "alert_type": "memory_high",
                "severity": "warning",
                "service": "data-sync-service",
                "value": "82.3%"
            },
            {
                "timestamp": "2026-04-25 10:22:10",
                "level": "WARNING",
                "message": "[告警] API 响应时间超过 2 秒",
                "alert_id": "ALT-003",
                "alert_type": "latency_high",
                "severity": "warning",
                "service": "api-gateway-service",
                "value": "2150ms"
            },
            {
                "timestamp": "2026-04-25 10:25:33",
                "level": "INFO",
                "message": "[恢复] CPU 使用率已恢复正常",
                "alert_id": "ALT-001",
                "alert_type": "cpu_high",
                "severity": "info",
                "service": "data-sync-service",
                "value": "45.2%"
            },
            {
                "timestamp": "2026-04-25 10:30:00",
                "level": "ERROR",
                "message": "[告警] 数据库连接池耗尽",
                "alert_id": "ALT-004",
                "alert_type": "db_connection",
                "severity": "error",
                "service": "data-sync-service",
                "value": "pool_size=0"
            }
        ]
        return {
            "topic_id": topic_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "query": query,
            "limit": limit,
            "total": len(alert_logs),
            "logs": alert_logs,
            "took_ms": 35,
            "message": f"成功查询 {len(alert_logs)} 条告警日志"
        }
    elif topic_id == "topic-001":
        # 应用日志：动态生成 INFO 日志
        logs = []
        current_time_ms = start_ms
        count = 0
        max_logs_by_time = int((end_ms - start_ms) / (60 * 1000)) + 1
        actual_limit = min(limit, max_logs_by_time)

        # 按分钟生成日志
        while current_time_ms <= end_ms and count < actual_limit:
            log_time = datetime.fromtimestamp(current_time_ms / 1000)
            time_str = log_time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "timestamp": time_str,
                "level": "INFO",
                "message": "正在同步元数据……"
            }
            logs.append(log_entry)
            count += 1
            current_time_ms += 60 * 1000

        return {
            "topic_id": topic_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "query": query,
            "limit": limit,
            "total": len(logs),
            "logs": logs,
            "took_ms": 50,
            "message": f"成功查询 {len(logs)} 条应用日志"
        }
    else:
        # 未知 topic_id：返回错误
        return {
            "topic_id": topic_id,
            "start_time": start_ms,
            "end_time": end_ms,
            "query": query,
            "limit": limit,
            "total": 0,
            "logs": [],
            "took_ms": 0,
            "error": f"主题不存在: {topic_id}",
            "message": f"错误: 未找到主题 {topic_id}，请检查 topic_id 是否正确。可用的 topic_id: topic-001 (应用日志), topic-alerts (告警日志)"
        }


# ========== 入口点 ==========
if __name__ == "__main__":
    # 使用 streamable-http 传输协议运行 MCP 服务器
    # 监听地址：127.0.0.1:8003
    # 路径：/mcp
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8003, path="/mcp")
