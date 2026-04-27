"""FastAPI 应用入口模块

主应用程序，负责：
1. 配置应用生命周期（启动/关闭时的资源管理）
2. 注册所有 API 路由
3. 配置中间件（CORS 等）
4. 挂载静态文件服务
5. 启动 uvicorn ASGI 服务器
"""

# ========== FastAPI 核心依赖 ==========
from fastapi import FastAPI                      # FastAPI 应用类
from fastapi.middleware.cors import CORSMiddleware  # 跨域资源共享中间件
from fastapi.staticfiles import StaticFiles      # 静态文件服务（CSS/JS/图片等）
from fastapi.responses import FileResponse       # 文件响应，用于返回 HTML 页面

# ========== 标准库 ==========
from contextlib import asynccontextmanager  # 异步上下文管理器，用于生命周期管理
import os                                # 路径操作，判断文件是否存在等

# ========== 项目内部依赖 ==========
from app.config import config            # 全局配置对象
from loguru import logger                # 项目日志记录器
from app.api import chat, health, file, aiops  # API 路由模块（按功能分类）
from app.core.milvus_client import milvus_manager  # Milvus 向量数据库连接管理器


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期异步上下文管理器

    FastAPI 在应用启动时执行 yield 之前的代码，在关闭时执行 yield 之后的代码。
    这是在应用层面管理全局资源（数据库连接、缓存、线程池等）的推荐方式。

    注意：这是异步函数，asynccontextmanager 装饰器使其可作为异步上下文管理器使用。

    Args:
        app: FastAPI 应用实例，用于在生命周期钩子中访问应用状态
    """

    # ========== 启动阶段：初始化全局资源 ==========

    # 打印醒目的启动分隔线，便于在日志中识别应用启动位置
    logger.info("=" * 60)
    # f-string 中使用 emoji 让控制台日志更直观
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")
    # 根据 debug 模式显示环境类型
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")
    # 显示服务监听地址和 API 文档地址，方便开发者快速访问
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # 连接 Milvus 向量数据库
    # 这是启动阶段最关键的操作之一，如果连接失败应用应该无法正常工作
    logger.info("🔌 正在连接 Milvus...")
    milvus_manager.connect()
    logger.info("✅ Milvus 连接成功")

    logger.info("=" * 60)

    # yield 将控制权交给 FastAPI，此时应用正常运行，处理 HTTP 请求
    yield

    # ========== 关闭阶段：清理全局资源 ==========
    # 应用收到终止信号（SIGTERM/SIGINT）时，代码从这里继续执行

    logger.info("🔌 正在关闭 Milvus 连接...")
    milvus_manager.close()  # 优雅关闭连接，释放资源
    logger.info(f"👋 {config.app_name} 关闭")


# ========== 创建 FastAPI 应用实例 ==========
# FastAPI 会自动生成 OpenAPI（Swagger）文档，访问 /docs 可查看交互式 API 文档
app = FastAPI(
    title=config.app_name,                    # API 文档标题
    version=config.app_version,              # API 版本号
    description="基于 LangChain 的智能oncall运维系统",  # API 描述
    lifespan=lifespan                        # 关联生命周期管理器
)

# ========== 配置 CORS 中间件 ==========
# CORS（跨域资源共享）允许浏览器跨域请求，解决前端页面与 API 域名不一致的问题
# 例如：前端在 http://localhost:3000，API 在 http://localhost:9900
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允许所有来源（生产环境应限制为具体域名列表）
    allow_credentials=True,   # 允许携带认证信息（cookies、Authorization header）
    allow_methods=["*"],      # 允许所有 HTTP 方法（GET/POST/PUT/DELETE 等）
    allow_headers=["*"],     # 允许所有请求头
)

# ========== 注册 API 路由 ==========
# 路由按功能模块分组，通过 tags 在 API 文档中分类显示
app.include_router(health.router, tags=["健康检查"])          # 健康检查接口
app.include_router(chat.router, prefix="/api", tags=["对话"])  # 对话相关接口（添加 /api 前缀）
app.include_router(file.router, prefix="/api", tags=["文件管理"])  # 文件上传和索引管理
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])  # 智能运维诊断接口

# ========== 挂载静态文件服务 ==========
# 将 static 目录下的文件通过 /static 路径对外提供访问
# 例如：static/style.css 访问路径为 /static/style.css
static_dir = "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """根路径处理函数

    当用户访问网站根路径时：
    1. 如果 static/index.html 存在，返回该 HTML 文件（单页应用入口）
    2. 否则返回 JSON 格式的欢迎信息

    这使得同一个服务既能提供 Web 界面，又能提供 API 服务。

    Returns:
        FileResponse: HTML 文件响应或包含版本信息的字典
    """
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        # FileResponse 自动设置正确的 Content-Type（text/html）和缓存头
        return FileResponse(index_path)
    # 兜底响应，确保 API 始终可用
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


# ========== 直接运行入口 ==========
# 当通过 python app/main.py 或 python -m app.main 启动时执行
# 不同于 uvicorn 直接启动，这种方式能让我们在启动前做一些预处理
if __name__ == "__main__":
    import uvicorn  # 延迟导入，仅在直接运行时才加载 uvicorn

    # uvicorn.run() 是启动 ASGI 服务器的便捷方法
    uvicorn.run(
        "app.main:app",           # 应用模块路径（模块名:应用实例名）
        host=config.host,         # 监听地址
        port=config.port,         # 监听端口
        reload=config.debug,      # 开发模式下启用代码热重载（自动重启）
        log_level="info"          # 日志级别
    )
