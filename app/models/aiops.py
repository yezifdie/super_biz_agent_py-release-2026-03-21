"""AIOps 智能运维数据模型模块

定义智能运维（Artificial Intelligence for IT Operations）场景下的数据模型。
AIOps 使用机器学习和大数据分析来自动化 IT 运维任务，如异常检测、根因分析和故障诊断。

本模块包含：
- AIOpsRequest: 运维诊断请求
- AlertInfo: 告警信息（从监控系统获取的告警数据）
- DiagnosisResponse: 诊断结果响应
"""

from typing import Optional, List, Dict, Any  # Optional: 可选字段，List: 列表，Dict: 字典，Any: 任意类型
from pydantic import BaseModel, Field  # BaseModel: 模型基类，Field: 字段元数据


class AIOpsRequest(BaseModel):
    """AIOps 智能运维诊断请求模型

    用户发起一次运维诊断时的请求格式。
    与普通对话不同，AIOps 请求会触发多步骤的自动化诊断流程：
    1. 分析告警信息
    2. 调用 MCP 工具获取日志和监控数据
    3. 结合知识库生成诊断报告

    场景示例：
    - 告警触发后自动诊断："收到 CPU 告警，请诊断原因"
    - 人工发起诊断："帮我分析为什么服务响应变慢了"
    - 批量诊断："对过去 1 小时的所有告警进行诊断"
    """

    # 会话 ID，用于追踪本次诊断的完整过程
    # 诊断过程可能是多轮的（规划→执行→重新规划），需要 session_id 串联
    # default="default" 表示如果不传，默认使用 "default" 会话
    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪诊断历史"
    )

    class Config:
        # 为 OpenAPI 文档提供请求示例
        json_schema_extra = {
            "example": {
                "session_id": "session-123"
            }
        }


class AlertInfo(BaseModel):
    """告警信息模型

    从监控系统（如 Prometheus、阿里云 ARMS）获取的告警数据结构。
    包含告警的名称、严重程度、影响的实例、持续时间等关键信息。

    字段对应关系：
    - alertname: 告警规则的名称，如 "HighCPUUsage"、"ServiceDown"
    - severity: 严重程度，常见值：critical（严重）、warning（警告）、info（信息）
    - instance: 触发告警的具体实例（主机名/IP/服务名）
    - duration: 告警持续时间，格式如 "5m"（5分钟）、"2h"（2小时）
    - description: 告警的详细描述，包含具体的指标数值和阈值

    示例告警：
        alertname: "HighCPUUsage"
        severity: "critical"
        instance: "api-server-01"
        duration: "10m"
        description: "CPU 使用率连续 10 分钟超过 90%"
    """

    # 告警规则名称，唯一标识一种告警类型
    # 用于后续的知识库检索："找到所有 HighCPUUsage 相关的处理方案"
    alertname: str

    # 告警严重程度等级
    # critical > warning > info
    # 不同等级可能触发不同的诊断策略和通知渠道
    severity: str

    # 触发告警的目标实例
    # 格式取决于监控系统，可能是 IP、主机名、服务名或容器 ID
    instance: str

    # 告警已经持续的时间
    # 格式：数字 + 单位，如 "5m"（分钟）、"2h"（小时）、"1d"（天）
    # 持续时间越长，通常问题越严重（系统未自动恢复）
    duration: str

    # 告警的详细描述，包含具体数值
    # 如："CPU 使用率 95%，阈值 80%"
    # 此字段可选，因为某些监控系统的告警摘要可能不包含详情
    description: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """AIOps 诊断结果响应模型（非流式）

    完整的诊断报告响应，包含诊断状态、告警信息、根因分析和处理建议。
    注意：本项目实际使用 SSE（Server-Sent Events）流式返回，
    此模型仅作为非流式响应的备用或用于结果缓存。

    流式响应 vs 非流式响应：
    - 流式：边诊断边返回，用户实时看到诊断进度（更好的 UX）
    - 非流式：等诊断完成一次性返回（便于缓存，但等待时间长）

    响应数据示例：
        {
            "code": 200,
            "message": "success",
            "data": {
                "status": "completed",
                "target_alert": {
                    "alertname": "HighCPUUsage",
                    "severity": "critical"
                },
                "diagnosis": {
                    "root_cause": "数据库慢查询导致连接池耗尽",
                    "recommendations": ["优化慢查询 SQL", "增加连接池大小"]
                }
            }
        }
    """

    # HTTP 状态码的逻辑码（区别于 HTTP 协议层面的状态码）
    # 200: 成功，400: 请求参数错误，500: 内部错误
    code: int = 200

    # 操作结果的描述信息
    message: str = "success"

    # 诊断结果数据，类型为字典，可包含任意结构
    # 典型结构：
    # {
    #     "status": "completed",           // completed / failed / in_progress
    #     "target_alert": {...},           // 目标告警信息
    #     "diagnosis": {
    #         "root_cause": "...",        // 根因分析
    #         "recommendations": [...]     // 建议措施
    #     }
    # }
    data: Dict[str, Any]

    class Config:
        # OpenAPI 文档示例
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "completed",
                    "target_alert": {
                        "alertname": "HighCPUUsage",
                        "severity": "critical"
                    },
                    "diagnosis": {
                        "root_cause": "数据库连接池耗尽",
                        "recommendations": ["扩容数据库连接池", "优化SQL查询"]
                    }
                }
            }
        }
