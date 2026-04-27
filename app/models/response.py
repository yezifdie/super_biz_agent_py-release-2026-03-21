"""响应数据模型模块

定义服务端返回给客户端的所有数据格式。
每个 Pydantic 模型对应一个 API 端点的响应格式。

设计原则：
- 所有响应都使用统一的数据结构（ApiResponse 包装）
- 包含足够的状态信息，方便前端判断请求是否成功
- 避免在响应中暴露敏感信息（如内部错误堆栈）
"""

# Pydantic 核心
from pydantic import BaseModel, Field  # BaseModel: 数据模型基类，Field: 字段元数据定义

# typing 模块用于类型注解
from typing import List, Dict, Any, Optional  # List: 列表类型，Dict: 字典类型，Any: 任意类型，Optional: 可选类型


class ChatResponse(BaseModel):
    """对话响应模型

    AI 对用户问题的回复响应。
    包含 AI 生成的回答内容和建议的会话 ID。

    前端接收示例（JSON）：
        {
            "answer": "向量数据库是一种专门用于存储和检索向量嵌入的数据库...",
            "session_id": "session-123"
        }
    """

    # AI 生成的回答文本
    # 可能包含 Markdown 格式（标题、代码块、列表等）
    # 长度取决于 LLM 模型输出，一般在几十到几千个字符之间
    answer: str = Field(..., description="AI 回答")

    # 本次对话的会话 ID
    # 前端应保存此 ID，用于后续消息的上下文关联
    # 如果传入的 ID 已被服务端存档，可能会返回新 ID
    session_id: str = Field(..., description="会话 ID")


class SessionInfoResponse(BaseModel):
    """会话信息响应模型

    返回指定会话的详细信息，包括消息历史。
    用于前端恢复历史对话或展示对话记录。

    场景：
    - 用户刷新页面后恢复对话
    - 展示会话列表中的对话详情
    - 导出对话记录
    """

    # 会话的唯一标识符
    session_id: str = Field(..., description="会话 ID")

    # 该会话中包含的消息总数
    # 可用于分页显示或判断对话长度
    message_count: int = Field(..., description="消息数量")

    # 历史消息列表，每条消息是一个字典
    # 字典结构：{"role": "user/assistant", "content": "消息内容"}
    # 按时间顺序排列，第一条是最早的消息
    # 注意：敏感信息（如 API 密钥）应在此处过滤掉
    history: List[Dict[str, str]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应包装模型

    这是一个统一的响应包装器，将所有 API 响应规范化为相同格式。
    前端可以统一处理：检查 status 字段判断成功/失败，读取 data 获取实际数据。

    设计意义：
    - 统一成功/失败响应的数据结构
    - 提供明确的操作结果反馈
    - 便于前端编写统一的响应处理逻辑

    前端接收示例（JSON）：
        {
            "status": "success",
            "message": "文件上传成功",
            "data": {
                "file_id": "file-456",
                "chunks_indexed": 15
            }
        }

        // 失败响应示例
        {
            "status": "error",
            "message": "文件大小超过限制",
            "data": null
        }
    """

    # 操作结果状态，可选值："success" | "error" | "warning"
    # 前端应据此显示成功提示或错误弹窗
    status: str = Field(..., description="状态")

    # 对操作结果的描述性信息
    # 成功时可能是"操作完成"，失败时是具体的错误原因
    message: str = Field(..., description="消息")

    # 实际返回的数据，类型可以是任意 JSON 兼容类型
    # 成功时包含业务数据（如文件信息、统计数据等）
    # 失败时为 None（Optional[Any]）
    # 使用 Any 而非具体类型是为了保持最大灵活性
    data: Optional[Any] = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应模型

    用于服务健康检查端点的响应格式。
    通常由负载均衡器或监控系统定期调用，以判断服务是否存活。

    判断规则：
    - status == "healthy" 且所有服务组件正常 → 服务健康
    - status == "unhealthy" → 服务异常，需要人工介入
    - status == "degraded" → 部分功能可用（如数据库正常但缓存异常）
    """

    # 服务整体状态：healthy / unhealthy / degraded
    status: str = Field(..., description="状态")

    # 服务名称标识，用于多服务环境下区分来源
    service: str = Field(..., description="服务名称")

    # 服务版本号
    # 用于排查问题时确认实际运行的版本
    # 如果版本不匹配，可能是部署未生效
    version: str = Field(..., description="版本号")
