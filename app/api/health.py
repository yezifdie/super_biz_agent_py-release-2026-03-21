"""健康检查接口模块

提供服务健康状态检查的 API。
用于：
1. 负载均衡器检测服务可用性
2. 监控系统检查组件状态
3. 运维人员排查问题

健康检查项目：
- 服务基本信息（名称、版本）
- Milvus 向量数据库连接状态

返回状态码：
- 200: 服务健康
- 503: 服务不可用（Milvus 连接失败）
"""

from typing import Any  # 类型注解
from fastapi import APIRouter  # FastAPI 路由
from fastapi.responses import JSONResponse  # JSON 响应
from app.config import config  # 全局配置
from app.core.milvus_client import milvus_manager  # Milvus 连接管理器
from loguru import logger  # 日志记录器

# 创建 APIRouter
router = APIRouter()


@router.get("/health")
async def health_check():
    """健康检查接口

    检查服务及其依赖组件的健康状态。
    建议每 30-60 秒调用一次，用于监控告警。

    请求示例：
        GET /api/health

    响应示例（健康）：
        HTTP 200
        {
            "code": 200,
            "message": "服务运行正常",
            "data": {
                "service": "SuperBizAgent",
                "version": "1.0.0",
                "status": "healthy",
                "milvus": {
                    "status": "connected",
                    "message": "Milvus 连接正常"
                }
            }
        }

    响应示例（不健康）：
        HTTP 503
        {
            "code": 503,
            "message": "服务不可用",
            "data": {
                "service": "SuperBizAgent",
                "version": "1.0.0",
                "status": "unhealthy",
                "error": "数据库不可用",
                "milvus": {
                    "status": "disconnected",
                    "message": "Milvus 连接异常"
                }
            }
        }

    Returns:
        JSONResponse: 健康检查结果
    """
    # 初始化健康数据
    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }

    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status: str = "connected" if milvus_healthy else "disconnected"
        milvus_message: str = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    # 判断整体健康状态
    # 如果 Milvus 不可用，整体服务不可用
    overall_status = "healthy"
    status_code = 200

    if health_data["milvus"]["status"] != "connected":
        overall_status = "unhealthy"
        status_code = 503
        health_data["error"] = "数据库不可用"

    health_data["status"] = overall_status

    # 返回 JSON 响应
    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "服务运行正常" if overall_status == "healthy" else "服务不可用",
            "data": health_data
        }
    )
