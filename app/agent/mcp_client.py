"""MCP（Model Context Protocol）客户端管理模块

MCP 是一个开放协议，用于将 AI 模型连接到外部工具和数据源。
本模块封装了 MCP 客户端的创建和管理，提供：
1. 多服务器 MCP 客户端（MultiServerMCPClient）
2. 自动重试拦截器（处理网络抖动和服务暂时不可用）
3. 全局单例管理（避免重复初始化）

核心概念：
- MCP Client: 连接到 MCP 服务器的客户端，负责工具发现和调用
- MCP Server: 提供工具的服务端（如日志查询、监控数据查询）
- Tool（工具）: MCP 服务器暴露的可调用函数（如 search_log、query_cpu_metrics）
- Interceptor（拦截器）: 在工具调用前后执行的钩子函数，用于日志、重试等

MCP 服务器配置：
- cls（日志服务）: http://localhost:8003/mcp
- monitor（监控服务）: http://localhost:8004/mcp

重试策略：
- 使用指数退避（Exponential Backoff）：1s → 2s → 4s
- 最多重试 3 次
- 重试失败后返回包含错误信息的 CallToolResult，而非抛出异常
"""

import asyncio  # 异步睡眠，用于重试延迟
from typing import Optional, Dict, Any, List  # 类型注解
from langchain_mcp_adapters.client import MultiServerMCPClient  # LangChain MCP 客户端
from langchain_mcp_adapters.interceptors import MCPToolCallRequest  # MCP 工具调用请求
from mcp.types import CallToolResult, TextContent  # MCP 类型定义
from loguru import logger  # 日志记录器

# 全局 MCP 客户端实例（延迟初始化，避免应用启动时就连接）
# 设为 None 表示尚未初始化，首次使用时才创建
_mcp_client: Optional[MultiServerMCPClient] = None


async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """MCP 工具调用重试拦截器

    拦截每个 MCP 工具调用，当调用失败时自动重试。
    采用指数退避策略，避免在服务端过载时雪上加霜。

    指数退避（Exponential Backoff）原理：
    - 第 1 次失败：等待 1 * 2^0 = 1 秒后重试
    - 第 2 次失败：等待 1 * 2^1 = 2 秒后重试
    - 第 3 次失败：等待 1 * 2^2 = 4 秒后重试

    这种策略的好处：
    1. 瞬时故障（如网络抖动）可以自动恢复
    2. 服务端有缓冲时间处理积压请求
    3. 避免大量客户端同时重试造成"惊群效应"

    MCPToolCallRequest 结构：
    - name: 工具名称，如 "search_log"
    - args: 工具参数字典，如 {"service": "api-server", "limit": 10}
    - server_name: 服务器名称，如 "cls"

    Args:
        request: MCP 工具调用请求，包含工具名、参数、服务器名
        handler: 实际的工具调用处理器（由 MCP 客户端提供）
        max_retries: 最大重试次数，默认 3 次
        delay: 初始延迟时间（秒），默认 1 秒

    Returns:
        CallToolResult: 工具调用结果
                        - 成功时：isError=False，content 包含结果
                        - 失败时：isError=True，content 包含错误信息

    设计决策：
    - 返回 CallToolResult 而非抛出异常，这样调用方可以统一处理成功和失败
    - 错误信息会被记录到日志，供运维人员排查
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            # 记录尝试次数和目标工具
            logger.info(
                f"调用 MCP 工具: {request.name} "
                f"(服务器: {request.server_name}, 第 {attempt + 1}/{max_retries} 次尝试)"
            )
            # 调用实际的工具处理器
            result = await handler(request)
            logger.info(f"MCP 工具 {request.name} 调用成功")
            return result

        except Exception as e:
            last_error = e
            logger.warning(
                f"MCP 工具 {request.name} 调用失败 "
                f"(第 {attempt + 1}/{max_retries} 次): {str(e)}"
            )

            # 如果不是最后一次尝试，等待后重试
            if attempt < max_retries - 1:
                # 指数退避：delay * 2^attempt
                # attempt=0: 1s, attempt=1: 2s, attempt=2: 4s
                wait_time = delay * (2 ** attempt)
                logger.info(f"等待 {wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)

    # 所有重试都失败，返回包含错误信息的 CallToolResult
    error_msg = f"工具 {request.name} 在 {max_retries} 次重试后仍然失败: {str(last_error)}"
    logger.error(error_msg)
    return CallToolResult(
        content=[TextContent(type="text", text=error_msg)],
        isError=True  # 标记为错误，让调用方知道调用失败了
    )


# 从配置文件读取 MCP 服务器配置
# DEFAULT_MCP_SERVERS 会在模块加载时从 config.mcp_servers 获取值
from app.config import config

# 将配置文件中的 MCP 服务器映射传递给 MultiServerMCPClient
# 格式：{"cls": {"transport": "streamable-http", "url": "http://..."}, ...}
DEFAULT_MCP_SERVERS = config.mcp_servers


async def get_mcp_client(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """获取或初始化 MCP 客户端（不带重试拦截器）

    单例模式：除非显式请求新实例（force_new=True），
    否则整个应用共享同一个 MCP 客户端。

    从 langchain-mcp-adapters 0.1.0 开始：
    - MultiServerMCPClient 不再支持作为上下文管理器
    - 直接创建实例即可使用
    - 不需要调用 __aenter__()

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
                 格式：{server_name: {"transport": "...", "url": "..."}}
        tool_interceptors: 自定义工具拦截器列表
                          拦截器会在工具调用前后执行，可用于日志、重试、监控等
        force_new: 是否强制创建新实例
                   正常情况下应使用 False（共享单例）
                   True 用于特殊场景：如需要不同的 MCP 服务器配置

    Returns:
        MultiServerMCPClient: MCP 客户端实例
                              可通过 client.get_tools() 获取可用工具列表
                              可通过 client.call_tool() 调用工具
    """
    global _mcp_client

    # 如果请求新实例，直接创建并返回（不缓存）
    if force_new:
        logger.info("创建新的 MCP 客户端实例（非单例）")
        client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors
        )
        # 不再需要 __aenter__()，直接返回即可
        return client

    # 单例模式：如果已存在，直接返回
    if _mcp_client is None:
        logger.info("初始化全局 MCP 客户端...")
        _mcp_client = _create_mcp_client(
            servers or DEFAULT_MCP_SERVERS,
            tool_interceptors
        )
        logger.info("全局 MCP 客户端初始化完成")

    return _mcp_client


async def get_mcp_client_with_retry(
    servers: Optional[Dict[str, Dict[str, str]]] = None,
    tool_interceptors: Optional[List] = None,
    force_new: bool = False
) -> MultiServerMCPClient:
    """获取或初始化带重试功能的 MCP 客户端

    与 get_mcp_client() 相同，但自动添加了 retry_interceptor 重试拦截器。
    重试拦截器会拦截所有工具调用，在失败时自动重试。

    这是推荐使用的方式，因为 MCP 服务可能因为网络问题暂时不可用。

    重试拦截器特点：
    1. 位于拦截器列表的最前面（最先执行）
    2. 指数退避策略：1s → 2s → 4s
    3. 最多重试 3 次
    4. 重试失败后返回错误结果而非抛出异常

    Args:
        servers: MCP 服务器配置，默认使用 DEFAULT_MCP_SERVERS
        tool_interceptors: 自定义拦截器（会添加在重试拦截器之后）
        force_new: 是否强制创建新实例

    Returns:
        MultiServerMCPClient: 带重试功能的 MCP 客户端实例
    """
    # 构建拦截器列表：重试拦截器在最前面
    interceptors = [retry_interceptor]
    if tool_interceptors:
        # 扩展列表：重试拦截器 + 自定义拦截器
        interceptors.extend(tool_interceptors)

    return await get_mcp_client(
        servers=servers,
        tool_interceptors=interceptors,
        force_new=force_new
    )


def _create_mcp_client(
    servers: Dict[str, Dict[str, str]],
    tool_interceptors: Optional[List] = None
) -> MultiServerMCPClient:
    """创建 MCP 客户端实例（内部方法）

    这是实际的客户端创建逻辑，被 get_mcp_client 和 get_mcp_client_with_retry 调用。

    MultiServerMCPClient 构造函数参数：
    - 第一个位置参数：servers 配置字典
    - tool_interceptors: 工具拦截器列表（可选）

    Args:
        servers: MCP 服务器配置字典
        tool_interceptors: 工具拦截器列表

    Returns:
        MultiServerMCPClient: 客户端实例（尚未连接，实际连接在首次调用工具时发生）
    """
    # 构建传递给 MultiServerMCPClient 的关键字参数
    kwargs: Dict[str, Any] = {}

    if tool_interceptors:
        kwargs["tool_interceptors"] = tool_interceptors

    # 创建并返回客户端实例
    # 第一个参数是 servers 配置字典
    return MultiServerMCPClient(servers, **kwargs)  # type: ignore[arg-type]
