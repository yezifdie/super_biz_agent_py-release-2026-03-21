"""AIOps 智能运维接口模块

提供基于 Plan-Execute-Replan 的智能故障诊断功能。
通过 SSE（Server-Sent Events）流式返回诊断过程和结果。

核心功能：
- 自动获取系统活动告警
- 制定诊断计划并逐步执行
- 查询日志和监控数据
- 生成结构化诊断报告

SSE 事件类型：
1. status: 状态更新（如"正在获取告警信息"）
2. plan: 诊断计划已制定
3. step_complete: 步骤执行完成
4. report: 最终诊断报告
5. complete: 诊断完成
6. error: 错误信息
"""

import json  # JSON 序列化
from fastapi import APIRouter  # FastAPI 路由
from sse_starlette.sse import EventSourceResponse  # SSE 响应
from loguru import logger  # 日志记录器

from app.models.aiops import AIOpsRequest  # 请求模型
from app.services.aiops_service import aiops_service  # AIOps 服务

# 创建 APIRouter
router = APIRouter()


@router.post("/aiops")
async def diagnose_stream(request: AIOpsRequest):
    """AIOps 故障诊断接口（流式 SSE）

    这是一个智能运维诊断接口，自动化执行以下流程：
    1. 获取系统活动告警
    2. 分析告警原因
    3. 制定诊断计划
    4. 执行诊断步骤（查询日志、监控数据等）
    5. 生成诊断报告

    使用 SSE 流式返回诊断过程，客户端可以实时看到诊断进度。

    请求示例：
        POST /api/aiops
        {
            "session_id": "session-123"
        }

    响应：SSE 事件流，data 字段为 JSON

    --- SSE 事件类型详解 ---

    1. status - 状态更新
    ```json
    {
        "type": "status",
        "stage": "fetching_alerts",
        "message": "正在获取系统告警信息..."
    }
    ```

    2. plan - 诊断计划制定完成
    ```json
    {
        "type": "plan",
        "stage": "plan_created",
        "message": "诊断计划已制定，共 6 个步骤",
        "target_alert": {...},
        "plan": ["步骤1: 查询告警详情", "步骤2: 分析日志", ...]
    }
    ```

    3. step_complete - 步骤执行完成
    ```json
    {
        "type": "step_complete",
        "stage": "step_executed",
        "message": "步骤执行完成 (2/6)",
        "current_step": "查询系统日志",
        "result_preview": "发现 3 条 ERROR 日志...",
        "remaining_steps": 4
    }
    ```

    4. report - 最终诊断报告
    ```json
    {
        "type": "report",
        "stage": "final_report",
        "message": "最终诊断报告已生成",
        "report": "# 故障诊断报告\n...",
        "evidence": {...}
    }
    ```

    5. complete - 诊断完成
    ```json
    {
        "type": "complete",
        "stage": "diagnosis_complete",
        "message": "诊断流程完成",
        "diagnosis": {
            "status": "completed",
            "report": "# 故障诊断报告\n..."
        }
    }
    ```

    6. error - 错误信息
    ```json
    {
        "type": "error",
        "stage": "error",
        "message": "诊断过程发生错误: ..."
    }
    ```

    --- curl 使用示例 ---
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \
      -H "Content-Type: application/json" \
      -d '{"session_id": "session-123"}' \
      --no-buffer
    ```

    --- 前端使用示例 ---
    ```javascript
    const response = await fetch('/api/aiops', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: 'session-123'})
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const {done, value} = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // SSE 事件格式: "data: {...}\n\n"
        const lines = text.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                handleEvent(data);
            }
        }
    }
    ```

    Args:
        request: AIOpsRequest，包含会话 ID

    Returns:
        EventSourceResponse: SSE 事件流
    """
    session_id = request.session_id or "default"
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式）")

    async def event_generator():
        """SSE 事件生成器

        异步生成诊断过程中的事件。
        每个事件都被序列化为 JSON 字符串并推送。
        """
        try:
            # 调用 AIOps 服务的诊断方法
            # diagnose() 是一个 AsyncGenerator，逐步返回诊断事件
            async for event in aiops_service.diagnose(session_id=session_id):
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())
