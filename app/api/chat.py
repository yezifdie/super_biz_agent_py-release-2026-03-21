"""对话接口模块

提供基于 RAG Agent 的对话功能，包括：
1. 普通对话（非流式，一次性返回结果）
2. 流式对话（SSE，边生成边返回）
3. 清空会话历史
4. 获取会话信息

SSE（Server-Sent Events）：
- 一种服务器推送技术，服务器可以向浏览器推送消息
- 适用于：实时聊天、进度更新、长时间任务通知
- 与 WebSocket 的区别：SSE 是单向的（服务端→客户端），更轻量
"""

import json                        # JSON 序列化
from fastapi import APIRouter, HTTPException  # FastAPI 路由和异常处理
from sse_starlette.sse import EventSourceResponse  # SSE 响应
from app.models.request import ChatRequest, ClearRequest  # 请求模型
from app.models.response import SessionInfoResponse, ApiResponse  # 响应模型
from app.services.rag_agent_service import rag_agent_service  # RAG Agent 服务
from loguru import logger  # 日志记录器

# 创建 APIRouter 实例
# Router 用于分组管理相关接口
router = APIRouter()


@router.post("/chat")
async def chat(request: ChatRequest):
    """快速对话接口（非流式）

    接收用户问题，返回 AI 生成的完整回答。
    适用于简单的一问一答场景。

    请求示例：
        POST /api/chat
        {
            "Id": "session-123",
            "Question": "CPU 使用率过高怎么办？"
        }

    响应示例：
        {
            "code": 200,
            "message": "success",
            "data": {
                "success": true,
                "answer": "当 CPU 使用率过高时，可以尝试以下方法...",
                "errorMessage": null
            }
        }

    Args:
        request: ChatRequest，包含会话 ID 和用户问题

    Returns:
        dict: 统一格式的 JSON 响应
    """
    try:
        logger.info(f"[会话 {request.id}] 收到快速对话请求: {request.question}")

        # 调用 RAG Agent 服务处理对话
        # query() 是非流式方法，等待完整回答后返回
        answer = await rag_agent_service.query(
            request.question,
            session_id=request.id
        )

        logger.info(f"[会话 {request.id}] 快速对话完成")

        # 返回统一格式的响应
        return {
            "code": 200,
            "message": "success",
            "data": {
                "success": True,
                "answer": answer,           # AI 生成的完整回答
                "errorMessage": None        # 错误信息（成功时为 null）
            }
        }

    except Exception as e:
        logger.error(f"对话接口错误: {e}")
        return {
            "code": 500,
            "message": "error",
            "data": {
                "success": False,
                "answer": None,
                "errorMessage": str(e)  # 错误信息
            }
        }


@router.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    """流式对话接口（SSE）

    使用 Server-Sent Events 向客户端推送流式响应。
    客户端可以实时看到 AI 逐步生成的回答。

    SSE 事件格式：
    - event: 固定为 "message"
    - data: JSON 字符串，包含事件类型和数据

    事件类型：
    1. content: AI 生成的内容片段
    2. tool_call: 工具调用开始/结束
    3. search_results: 检索结果
    4. done: 生成完成
    5. error: 错误信息
    6. debug: 调试信息

    前端使用示例：
    ```javascript
    const eventSource = new EventSource('/api/chat_stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({Id: 'session-123', Question: '问题'})
    });

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'content') {
            // 追加内容到聊天框
            appendToChat(data.data);
        } else if (data.type === 'done') {
            // 生成完成
            eventSource.close();
        }
    };
    ```

    Args:
        request: ChatRequest，包含会话 ID 和用户问题

    Returns:
        EventSourceResponse: SSE 事件流
    """
    logger.info(f"[会话 {request.id}] 收到流式对话请求: {request.question}")

    async def event_generator():
        """SSE 事件生成器

        异步生成器，每次 yield 一个事件。
        FastAPI 会将这些事件逐个推送给客户端。
        """
        try:
            # 调用 RAG Agent 的流式方法
            async for chunk in rag_agent_service.query_stream(
                request.question,
                session_id=request.id
            ):
                chunk_type = chunk.get("type", "unknown")
                chunk_data = chunk.get("data", None)

                # 根据事件类型生成不同的 SSE 事件
                if chunk_type == "debug":
                    # 调试信息（可选发送）
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "debug",
                            "node": chunk.get("node", "unknown"),
                            "message_type": chunk.get("message_type", "unknown")
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "tool_call":
                    # 工具调用事件
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "tool_call",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "search_results":
                    # 检索结果事件
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "search_results",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "content":
                    # 内容片段（最常见的事件）
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "content",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "complete":
                    # 完成信号
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "done",
                            "data": chunk_data
                        }, ensure_ascii=False)
                    }
                elif chunk_type == "error":
                    # 错误信息
                    yield {
                        "event": "message",
                        "data": json.dumps({
                            "type": "error",
                            "data": str(chunk_data)
                        }, ensure_ascii=False)
                    }

            logger.info(f"[会话 {request.id}] 流式对话完成")

        except Exception as e:
            logger.error(f"流式对话接口错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "data": str(e)
                }, ensure_ascii=False)
            }

    # 返回 SSE 响应
    # EventSourceResponse 会自动处理 SSE 协议
    return EventSourceResponse(event_generator())


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_session(request: ClearRequest):
    """清空会话历史接口

    清除指定会话的所有消息历史。
    通常在用户点击"新对话"按钮时调用。

    请求示例：
        POST /api/chat/clear
        {
            "sessionId": "session-123"
        }

    响应示例：
        {
            "status": "success",
            "message": "会话已清空",
            "data": null
        }

    Args:
        request: ClearRequest，包含要清除的会话 ID

    Returns:
        ApiResponse: 操作结果

    Raises:
        HTTPException: 500 错误时抛出
    """
    try:
        # 调用 RAG Agent 清除会话
        success = rag_agent_service.clear_session(request.session_id)
        logger.info(f"清空会话: {request.session_id}, 结果: {success}")

        return ApiResponse(
            status="success" if success else "error",
            message="会话已清空" if success else "清空会话失败",
            data=None
        )

    except Exception as e:
        logger.error(f"清空会话错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/session/{session_id}", response_model=SessionInfoResponse)
async def get_session_info(session_id: str) -> SessionInfoResponse:
    """获取会话历史接口

    查询指定会话的消息历史，用于恢复对话或展示对话记录。

    请求示例：
        GET /api/chat/session/session-123

    响应示例：
        {
            "session_id": "session-123",
            "message_count": 6,
            "history": [
                {"role": "user", "content": "问题1", "timestamp": "..."},
                {"role": "assistant", "content": "回答1", "timestamp": "..."}
            ]
        }

    Args:
        session_id: 会话 ID（路径参数）

    Returns:
        SessionInfoResponse: 会话信息，包含消息历史

    Raises:
        HTTPException: 500 错误时抛出
    """
    try:
        # 获取会话历史
        history = rag_agent_service.get_session_history(session_id)

        return SessionInfoResponse(
            session_id=session_id,
            message_count=len(history),
            history=history
        )

    except Exception as e:
        logger.error(f"获取会话信息错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))
