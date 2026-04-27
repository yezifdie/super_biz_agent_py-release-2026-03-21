"""请求数据模型模块

定义所有客户端发送给服务端的数据结构。
每个 Pydantic 模型对应一个 API 端点的请求格式。
使用 Pydantic 可以：
1. 自动验证请求数据的类型和格式
2. 提供清晰的错误信息（告诉客户端哪里数据不对）
3. 自动生成 OpenAPI 文档中的请求示例
4. 在请求进入业务逻辑之前进行数据清洗
"""

from pydantic import BaseModel, Field  # BaseModel 是所有模型的基类，Field 用于自定义字段元数据


class ChatRequest(BaseModel):
    """对话请求模型

    用户发送聊天消息时的请求格式。
    包含会话 ID（用于维护对话上下文）和问题内容。

    前端发送示例（JSON）：
        {
            "Id": "session-123",
            "Question": "CPU 使用率过高怎么办？"
        }

    对应后端接收方式：
        @app.post("/api/chat")
        def chat(request: ChatRequest):
            ...
    """

    # 会话 ID，用于标识一次独立的对话会话
    # 同一个 session_id 的消息会被串联成对话上下文
    # Field(...) 中的 ... 表示该字段必填（不可为 None）
    # alias="Id" 允许前端使用驼峰命名 "Id"，后端转换为蛇形 "id"
    id: str = Field(..., description="会话 ID", alias="Id")

    # 用户输入的问题内容
    # 最长限制取决于前端和 LLM 的上下文窗口大小
    question: str = Field(..., description="用户问题", alias="Question")

    class Config:
        # populate_by_name = True：允许通过字段名或别名两种方式赋值
        # 即 request.id 和 request.Id 都能正确赋值
        populate_by_name = True

        # json_schema_extra：为 OpenAPI 文档提供示例数据
        # 这样在 /docs 页面可以看到请求的 JSON 示例
        json_schema_extra = {
            "example": {
                "Id": "session-123",
                "Question": "什么是向量数据库？"
            }
        }


class ClearRequest(BaseModel):
    """清空会话请求模型

    用户请求清除指定会话的历史记录时使用。
    通常在开始新对话或用户点击"清空对话"按钮时触发。

    作用：
    - 释放服务端存储的对话历史
    - 重置会话状态
    - 避免历史对话影响后续回答
    """

    # 要清除的会话 ID
    # 前端使用 "sessionId"（驼峰），后端接收时自动映射为 session_id
    session_id: str = Field(..., description="会话 ID", alias="sessionId")

    class Config:
        # 同上，允许字段名和别名两种方式赋值
        populate_by_name = True
