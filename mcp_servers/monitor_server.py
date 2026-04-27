"""智能运维监控 MCP Server

本地实现的监控服务 MCP Server，提供：
- 监控数据查询（CPU、内存、磁盘、网络等）
- 进程信息查询
- 历史工单查询
- 服务信息查询

用于支持运维 Agent 的故障排查场景。
运行在独立进程中（端口 8004）。

MCP Server 架构：
- 使用 FastMCP 框架构建
- 暴露多个工具函数供 Agent 调用
- 主应用通过 langchain-mcp-adapters 连接

提供的工具：
- query_cpu_metrics: 查询 CPU 使用率监控数据
- query_memory_metrics: 查询内存使用监控数据

注意：当前实现使用 Mock 数据，实际生产环境需要连接真实的监控系统 API。
"""

import logging          # 标准库日志模块
import functools        # 函数装饰器工具
import json            # JSON 序列化
import random          # 随机数生成（用于 Mock 数据）
from typing import Dict, Any, Optional  # 类型注解
from datetime import datetime, timedelta  # 日期时间处理
from fastmcp import FastMCP  # FastMCP 框架

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor_MCP_Server")

# 创建 FastMCP 实例
mcp = FastMCP("Monitor")


def log_tool_call(func):
    """工具调用日志装饰器

    记录每个工具函数的调用信息，便于调试和问题排查。
    使用 functools.wraps 保持原函数的元数据。

    使用方式：
        @mcp.tool()
        @log_tool_call
        def my_tool(...):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数
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

            # 记录返回结果摘要
            if isinstance(result, dict):
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


# ============================================================
# 辅助函数
# ============================================================

def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）
                              正数：未来
                              负数：过去

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # 返回默认时间（当前时间 + 偏移）
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成时间序列字符串

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量
        format_str: 时间格式字符串

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime(format_str)


# ============================================================
# 监控数据查询工具
# ============================================================

@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的 CPU 使用率监控数据

    返回指定时间范围内的 CPU 使用率数据点序列，以及统计信息和告警状态。

    参数说明：
    - service_name: 要查询的服务名称
    - start_time/end_time: 时间范围（字符串格式）
    - interval: 数据聚合间隔（1m/5m/1h）

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"
        start_time: 开始时间（可选）
            格式: "YYYY-MM-DD HH:MM:SS"
            默认: 当前时间的 1 小时前
        end_time: 结束时间（可选）
            格式: "YYYY-MM-DD HH:MM:SS"
            默认: 当前时间
        interval: 数据聚合间隔（默认 "1m"）
            - "1m": 每分钟一个数据点
            - "5m": 每 5 分钟一个数据点
            - "1h": 每小时一个数据点

    Returns:
        Dict: CPU 监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (cpu_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表
                * timestamp: 时间点（格式: HH:MM）
                * value: CPU 使用率百分比
                * process_id: 进程 ID
            - statistics: 统计信息
                * avg: 平均值
                * max: 最大值
                * min: 最小值
                * p95: 95 分位数
                * spike_detected: 是否检测到突增
            - alert_info: 告警信息
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息

    使用示例:
        # 使用默认时间（最近1小时）
        query_cpu_metrics(service_name="data-sync-service")

        # 指定时间范围
        query_cpu_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )
    """
    # 解析时间参数
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)

    # 解析间隔时间（interval: 1m, 5m, 1h 等）
    interval_minutes = 1  # 默认 1 分钟
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    # 动态生成 CPU 使用率数据（模拟从低到高逐渐增长）
    data_points = []
    current_time = start_dt
    time_index = 0

    # 初始 CPU 使用率（10%）
    base_cpu = 10.0

    while current_time <= end_dt:
        # CPU 使用率逐渐升高的算法：
        # - 前几个数据点保持在 10% 左右（正常状态）
        # - 然后开始快速上升（异常状态）
        # - 最终达到 95% 左右（触发告警）

        if time_index < 3:
            # 初始阶段：10% 左右波动
            cpu_value = base_cpu + (time_index * 0.5)
        else:
            # 上升阶段：使用指数增长模型
            growth_factor = (time_index - 2) * 8.5
            cpu_value = min(base_cpu + growth_factor, 96.0)

        # 添加随机波动（±2%）
        cpu_value = round(cpu_value + random.uniform(-2, 2), 1)
        cpu_value = max(0, min(100, cpu_value))  # 确保在 0-100 范围内

        data_point = {
            "timestamp": current_time.strftime("%H:%M"),
            "value": cpu_value,
            "process_id": "pid-12345"
        }

        data_points.append(data_point)

        # 下一个时间点
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    # 计算统计信息
    if data_points:
        values = [d["value"] for d in data_points]
        avg_value = round(sum(values) / len(values), 2)
        max_value = max(values)
        min_value = min(values)

        # 检测是否有 CPU 突增（超过 80%）
        spike_detected = max_value > 80.0

        return {
            "service_name": service_name,
            "metric_name": "cpu_usage_percent",
            "interval": interval,
            "data_points": data_points,
            "statistics": {
                "avg": avg_value,
                "max": max_value,
                "min": min_value,
                "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else max_value, 2),
                "spike_detected": spike_detected
            },
            "alert_info": {
                "triggered": spike_detected,
                "threshold": 80.0,
                "message": "CPU 使用率持续超过 80% 阈值" if spike_detected else "CPU 使用率正常"
            }
        }
    else:
        return {
            "service_name": service_name,
            "metric_name": "cpu_usage_percent",
            "interval": interval,
            "data_points": [],
            "statistics": {},
        }


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的内存使用监控数据

    返回指定时间范围内的内存使用率数据点序列，以及统计信息和告警状态。

    Args:
        service_name: 服务名称（必填）
            示例: "data-sync-service"
        start_time: 开始时间（可选）
            格式: "YYYY-MM-DD HH:MM:SS"
            默认: 当前时间的 1 小时前
        end_time: 结束时间（可选）
            格式: "YYYY-MM-DD HH:MM:SS"
            默认: 当前时间
        interval: 数据聚合间隔（默认 "1m"）
            - "1m": 每分钟一个数据点
            - "5m": 每 5 分钟一个数据点
            - "1h": 每小时一个数据点

    Returns:
        Dict: 内存监控数据
            - service_name: 服务名称
            - metric_name: 指标名称 (memory_usage_percent)
            - interval: 数据聚合间隔
            - data_points: 数据点列表
                * timestamp: 时间点（格式: HH:MM）
                * value: 内存使用率百分比
                * used_gb: 已使用内存（GB）
                * total_gb: 总内存（GB）
            - statistics: 统计信息
                * avg: 平均值
                * max: 最大值
                * min: 最小值
                * p95: 95 分位数
                * memory_pressure: 是否存在内存压力
            - alert_info: 告警信息
                * triggered: 是否触发告警
                * threshold: 告警阈值
                * message: 告警消息

    使用示例:
        # 使用默认时间（最近1小时）
        query_memory_metrics(service_name="data-sync-service")

        # 指定时间范围
        query_memory_metrics(
            service_name="data-sync-service",
            start_time="2026-02-14 10:00:00",
            end_time="2026-02-14 11:00:00",
            interval="5m"
        )
    """
    # 解析时间参数
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)

    # 解析间隔时间
    interval_minutes = 1  # 默认 1 分钟
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    # 动态生成内存使用率数据（模拟从低到高逐渐增长）
    data_points = []
    current_time = start_dt
    time_index = 0

    # 初始内存使用率（30%），总内存 8GB
    base_memory = 30.0
    total_gb = 8.0

    while current_time <= end_dt:
        # 内存使用率逐渐升高的算法：
        # - 前几个数据点保持在 30% 左右（正常状态）
        # - 然后开始逐步上升（内存泄漏或负载增加）
        # - 最终达到 85% 左右（触发告警）
        # 注意：内存增长比 CPU 慢（线性增长）

        if time_index < 3:
            # 初始阶段：30% 左右波动
            memory_value = base_memory + (time_index * 1.0)
        else:
            # 上升阶段：使用线性增长模型
            growth_factor = (time_index - 2) * 5.5
            memory_value = min(base_memory + growth_factor, 85.0)

        # 添加随机波动（±1%）
        memory_value = round(memory_value + random.uniform(-1, 1), 1)
        memory_value = max(0, min(100, memory_value))  # 确保在 0-100 范围内

        # 计算已使用内存（GB）
        used_gb = round((memory_value / 100.0) * total_gb, 2)

        data_point = {
            "timestamp": current_time.strftime("%H:%M"),
            "value": memory_value,
            "used_gb": used_gb,
            "total_gb": total_gb
        }

        data_points.append(data_point)

        # 下一个时间点
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    # 计算统计信息
    if data_points:
        values = [d["value"] for d in data_points]
        avg_value = round(sum(values) / len(values), 2)
        max_value = max(values)
        min_value = min(values)

        # 检测是否有内存压力（超过 70%）
        memory_pressure = max_value > 70.0

        return {
            "service_name": service_name,
            "metric_name": "memory_usage_percent",
            "interval": interval,
            "data_points": data_points,
            "statistics": {
                "avg": avg_value,
                "max": max_value,
                "min": min_value,
                "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else max_value, 2),
                "memory_pressure": memory_pressure
            },
            "alert_info": {
                "triggered": memory_pressure,
                "threshold": 70.0,
                "message": "内存使用率超过 70% 阈值，存在内存压力" if memory_pressure else "内存使用率正常"
            }
        }
    else:
        return {
            "service_name": service_name,
            "metric_name": "memory_usage_percent",
            "interval": interval,
            "data_points": [],
            "statistics": {},
            "error": "时间范围无效或没有生成数据点"
        }


# ========== 入口点 ==========
if __name__ == "__main__":
    # 使用 streamable-http 传输协议运行 MCP 服务器
    # 监听地址：127.0.0.1:8004
    # 路径：/mcp
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
