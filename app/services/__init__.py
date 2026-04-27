"""服务层模块

本模块包含应用的核心业务逻辑服务。

服务层设计原则：
- 单一职责：每个服务类专注于一个业务领域
- 依赖注入：服务之间通过构造函数或全局单例相互依赖
- 延迟初始化：全局单例在模块加载时创建，但内部资源可能在首次使用时才初始化

服务列表：
- rag_agent_service: RAG 对话服务（基于 LangGraph 的智能对话代理）
- aiops_service: AIOps 智能运维服务（Plan-Execute-Replan 工作流）
- vector_store_manager: 向量存储管理器（Milvus 底层操作封装）
- vector_embedding_service: 文本嵌入服务（文本→向量转换）
- vector_index_service: 向量索引服务（文档读取、分块、存储）
- vector_search_service: 向量检索服务（相似度搜索）
- document_splitter_service: 文档分割服务（文本分块策略）
"""
