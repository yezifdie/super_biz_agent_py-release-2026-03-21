"""RAG Agent 服务模块

基于 LangGraph 和 langchain_qwq 的智能对话代理服务。
RAG = Retrieval-Augmented Generation（检索增强生成）

核心功能：
- 对话管理：维护多轮对话上下文
- 工具调用：自动调用知识库检索、获取时间等工具
- 流式输出：支持边生成边返回（更好的用户体验）
- 会话持久化：使用 MemorySaver 保存对话历史

技术架构：
- 框架：LangGraph（状态图工作流）
- 模型：阿里云 ChatQwen（langchain_qwq）
- 工具：LangChain Tool + MCP 适配器
- 会话：LangGraph MemorySaver（内存存储）

使用示例：
    rag_service = RagAgentService(streaming=True)
    answer = await rag_service.query("CPU 使用率过高怎么办？", session_id="session-123")
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

# LangChain Agent 创建函数
from langchain.agents import create_agent

# LangChain 消息类型
from langchain_core.messages import (
    BaseMessage,       # 消息基类
    HumanMessage,      # 用户消息
    RemoveMessage,     # 消息删除操作（用于修剪历史）
    SystemMessage,     # 系统提示词
)

# LangGraph 内存检查点（保存对话状态）
from langgraph.checkpoint.memory import MemorySaver

# LangGraph 消息操作（add_messages 用于状态更新）
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger

# TypedDict：类型安全的字典（用于定义 Agent 状态）
from typing_extensions import TypedDict

# 阿里云 Qwen 模型（LangChain 集成）
from langchain_qwq import ChatQwen

from app.config import config  # 全局配置
from app.tools import get_current_time, retrieve_knowledge  # 本地工具
from app.agent.mcp_client import get_mcp_client_with_retry  # MCP 客户端

# 文档参考：https://docs.langchain.com/oss/python/integrations/chat/qwen
# 环境变量：需要 DASHSCOPE_API_BASE 和 DASHSCOPE_API_KEY


class AgentState(TypedDict):
    """Agent 状态定义

    使用 TypedDict 定义 LangGraph Agent 的状态结构。
    LangGraph 会自动维护状态，并在节点之间传递。

    字段说明：
    - messages: 消息历史列表（Annotated 用于增量更新）
      使用 Annotated[Sequence[BaseMessage], add_messages]
      意味着新消息会被追加到列表，而非替换
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """修剪消息历史的中间件

    为了避免对话历史无限增长导致上下文溢出，
    在每次响应后修剪过长的消息历史。

    修剪策略：
    1. 保留第一条系统消息（SystemMessage）
       系统消息包含工具定义和角色设定，不能丢弃
    2. 保留最近 6 条消息（3 轮对话）
       确保 LLM 看到最近的上下文
    3. 当消息数量 <= 7 时，不做修剪
       短对话不需要修剪

    Args:
        state: Agent 状态（包含 messages 列表）

    Returns:
        dict | None: 包含修剪后消息的字典，如果无需修剪则返回 None
                    返回值会被 LangGraph 自动合并到状态中
    """
    messages = state["messages"]

    # 短对话不需要修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息（保留工具定义和角色设定）
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    # 偶数条消息取最近 6 条，奇数条取最近 7 条
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表：系统消息 + 最近的对话
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    # 返回更新：删除所有旧消息，添加新消息
    # RemoveMessage(id=REMOVE_ALL_MESSAGES) 删除所有现有消息
    # *new_messages 添加保留的消息
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务类

    智能对话代理，支持：
    - 多轮对话（通过 session_id 区分不同会话）
    - 工具调用（知识库检索、获取时间、MCP 工具）
    - 流式输出（边生成边返回）
    - 会话持久化（通过 MemorySaver）

    初始化流程：
    1. 创建 ChatQwen 模型实例
    2. 准备基础工具列表
    3. 创建 MemorySaver 检查点
    4. 异步初始化 MCP 工具

    使用方式：
        # 非流式（一次性获取完整回答）
        answer = await rag_service.query("问题", "session-123")

        # 流式（边生成边返回）
        async for chunk in rag_service.query_stream("问题", "session-123"):
            print(chunk["data"], end="")
    """

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出
                      True: 边生成边返回（响应快，用户体验好）
                      False: 等全部生成后一次性返回（便于处理）
        """
        self.model_name = config.rag_model      # LLM 模型名称
        self.streaming = streaming              # 是否流式输出
        self.system_prompt = self._build_system_prompt()  # 系统提示词

        # 创建 ChatQwen 模型实例
        # ChatQwen 是 langchain_qwq 提供的 LangChain 集成
        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,  # 0.7 是平衡值：既有创造性又不过于随机
            streaming=streaming,
        )

        # 基础工具列表（项目内部实现的工具）
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 工具列表（延迟初始化）
        self.mcp_tools: list = []

        # 创建内存检查点（用于会话持久化）
        # MemorySaver 将状态保存在内存中，应用重启后会话历史丢失
        # 生产环境可替换为 SQLiteSaver、PostgresSaver 等持久化存储
        self.checkpointer = MemorySaver()

        # Agent 实例（延迟初始化）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）

        在首次调用 query() 或 query_stream() 时自动调用。
        这样做的好处：
        1. 避免应用启动时阻塞
        2. MCP 服务可能未就绪，延迟初始化有更多准备时间
        """
        if self._agent_initialized:
            return

        # 获取 MCP 客户端（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具列表
        # MCP 工具包括：日志查询、监控数据查询等
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 保存到实例变量
        self.mcp_tools = mcp_tools

        # 合并所有工具：基础工具 + MCP 工具
        all_tools = self.tools + self.mcp_tools

        # 创建 LangChain Agent
        # create_agent() 创建的是 React 风格的 Agent
        # Agent 会自动：
        # 1. 理解用户问题
        # 2. 决定是否需要调用工具
        # 3. 调用工具获取信息
        # 4. 基于工具结果生成回答
        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True

        # 记录可用工具列表
        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """构建系统提示词

        系统提示词定义 Agent 的角色定位和行为规范。
        注意：LangChain Agent 会自动将工具信息传递给 LLM，
        因此系统提示词中不需要列举具体工具。

        Returns:
            str: 系统提示词文本
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。

            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """非流式处理用户问题（一次性返回完整答案）

        适用场景：
        - API 调用（不依赖实时流式传输）
        - 需要完整答案后再处理
        - 简单的一问一答场景

        Args:
            question: 用户的问题
            session_id: 会话 ID（用于维护对话上下文）

        Returns:
            str: LLM 生成的完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 构建消息列表：系统提示词 + 用户问题
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            # 同一个 session_id 的消息会被关联到同一个对话线程
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 执行 Agent
            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用（如果有）
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}")
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理用户问题（逐步返回答案片段）

        适用场景：
        - WebSocket 或 SSE 推送
        - 前端需要实时显示打字效果
        - 长回答需要边生成边展示

        Yields:
            Dict[str, Any]: 流式数据块
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容（文本或错误信息）
                - node: 来自的 LangGraph 节点名
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 构建消息列表
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            agent_input = {"messages": messages}

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # astream() 是异步流式执行
            # stream_mode="messages" 表示按消息块输出
            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                # 从 metadata 中提取节点名
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                # 只处理 AI 消息块
                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            # 处理文本块
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    def get_session_history(self, session_id: str) -> list:
        """获取会话历史

        从 MemorySaver 检查点中读取指定会话的消息历史。

        Args:
            session_id: 会话 ID（即 thread_id）

        Returns:
            list: 消息历史列表，格式为：
                  [{"role": "user"|"assistant", "content": "...", "timestamp": "..."}]
        """
        try:
            config = {"configurable": {"thread_id": session_id}}

            # 获取该 thread 的最新检查点
            checkpoint_tuple = self.checkpointer.get(config)

            if not checkpoint_tuple:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            # 安全提取 checkpoint 数据
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
            else:
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}

            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])

            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)

                # 提取时间戳
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史

        从 MemorySaver 中删除指定会话的所有消息。

        Args:
            session_id: 会话 ID（即 thread_id）

        Returns:
            bool: 是否成功清除
        """
        try:
            self.checkpointer.delete_thread(session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源

        释放 Agent 持有的资源。
        注意：MCP 客户端由全局管理器统一管理，这里不需要清理。
        """
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# ========== 全局 RAG Agent 服务单例 ==========
# 启用流式输出（默认）
rag_agent_service = RagAgentService(streaming=True)
