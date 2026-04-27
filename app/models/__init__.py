"""数据模型模块

本模块定义了应用中使用的所有 Pydantic 数据模型。
Pydantic 是一个数据验证库，通过 Python 类型注解自动验证 JSON/字典等数据的有效性。

模块组织结构：
- request.py: API 请求模型（客户端发给服务端的数据）
- response.py: API 响应模型（服务端返回给客户端的数据）
- aiops.py: AIOps 智能运维专用模型
- document.py: 文档处理相关模型

使用方式：
    from app.models import ChatRequest, ChatResponse
    request = ChatRequest.model_validate(json_data)
"""
