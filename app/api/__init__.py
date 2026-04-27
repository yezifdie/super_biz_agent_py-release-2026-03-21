"""API 路由模块

本模块定义了所有 HTTP API 端点。
使用 FastAPI Router 将相关接口分组，便于管理和维护。

API 路由列表：
- chat.py: 对话相关接口（RAG 聊天、流式对话、会话管理）
- aiops.py: AIOps 智能运维接口（故障诊断）
- file.py: 文件管理接口（上传、索引）
- health.py: 健康检查接口

设计原则：
- RESTful 风格：使用标准的 HTTP 方法（GET/POST/PUT/DELETE）
- 统一响应格式：所有接口返回相同格式的 JSON 响应
- 错误处理：异常被捕获并返回友好的错误信息
"""
