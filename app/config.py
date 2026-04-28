"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理。
所有配置项从 .env 环境变量文件加载，支持默认值、类型验证和敏感信息脱敏。
"""

# 标准库导入
from typing import Dict, Any  # Dict 用于 MCP 服务器配置的类型注解，Any 用于动态配置
from pydantic_settings import BaseSettings, SettingsConfigDict  # Pydantic v2 的配置管理核心类


class Settings(BaseSettings):
    """应用全局配置类

    继承自 Pydantic BaseSettings，从环境变量和 .env 文件中加载配置。
    Pydantic 自动进行类型验证、默认值填充和敏感信息处理。

    使用方式：
        from app.config import config
        print(config.dashscope_api_key)
    """

    # model_config 是 Pydantic v2 的配置字典，替代了原来的 Config 类
    model_config = SettingsConfigDict(
        env_file=".env",          # 指定 .env 文件路径（相对于项目根目录）
        env_file_encoding="utf-8", # .env 文件编码为 UTF-8，确保中文注释正确读取
        case_sensitive=False,      # 环境变量名不区分大小写（默认行为）
        extra="ignore",            # 忽略 .env 中未定义的额外字段，避免警告
    )

    # ========== 应用基础配置 ==========
    # 应用名称，用于文档标题和日志输出
    app_name: str = "SuperBizAgent"
    # 应用版本号，用于版本管理和 API 响应
    app_version: str = "1.0.0"
    # 调试模式开关，True 时启用详细日志和代码热重载
    debug: bool = False
    # FastAPI 服务监听地址，0.0.0.0 表示接受所有网络接口的连接
    host: str = "0.0.0.0"
    # FastAPI 服务监听端口
    port: int = 9900

    # ========== 阿里云 DashScope（大模型服务）配置 ==========
    # DashScope API 密钥，用于调用阿里云 Qwen 系列模型
    # 实际使用时必须从环境变量 DASHSCOPE_API_KEY 加载，否则为空字符串导致调用失败
    dashscope_api_key: str = ""
    # 默认使用的 DashScope 模型名称，可选：qwen-max、qwen-plus、qwq-32b 等
    # qwen-max：效果最好但速度较慢、成本较高
    # qwen-plus：性价比均衡
    dashscope_model: str = "qwen-max"
    # 文本嵌入模型，用于将文本转换为向量以便存储和检索
    # text-embedding-v4 支持多种维度（默认 1024 维），兼容性最好
    dashscope_embedding_model: str = "text-embedding-v4"

    # ========== Milvus 向量数据库配置 ==========
    # Milvus 是开源的向量数据库，用于存储和检索文档嵌入向量
    # 实际生产环境应部署为分布式集群以支持高并发
    milvus_host: str = "localhost"  # Milvus 服务地址
    milvus_port: int = 19530        # Milvus gRPC 端口（默认 19530）
    milvus_timeout: int = 10000     # 连接超时时间，单位毫秒（10秒）

    # ========== RAG（检索增强生成）配置 ==========
    # RAG = Retrieval-Augmented Generation，即检索增强生成
    # 通过从向量数据库中检索相关文档，再让 LLM 基于检索结果生成回答
    # 每次检索返回的最相关文档数量，数值越大上下文越长但消耗越多 token
    rag_top_k: int = 3
    # RAG 流程中使用的 LLM 模型
    # 使用不带扩展思考的模型可以加快响应速度、减少 token 消耗
    rag_model: str = "qwen-max"

    # ========== 混合检索配置 ==========
    # 混合检索 = BM25 关键词检索 + 向量语义检索 + RRF 融合
    # 详细说明见 app/services/hybrid_retriever.py
    use_hybrid_retrieval: bool = True        # 是否启用混合检索
    use_reranker: bool = True               # 是否使用重排序模型
    rrf_k: int = 60                        # RRF 融合平滑参数（通常 60）
    bm25_weight: float = 0.4                # BM25 检索权重 (bm25_weight + vector_weight = 1.0)
    vector_weight: float = 0.6              # 向量检索权重
    rerank_top_n: int = 20                 # 重排序时取前 N 个结果

    # ========== 文档分块配置 ==========
    # 将长文档切割成小块时的参数配置
    # 分块策略直接影响 RAG 检索质量：块太小丢失上下文，块太大引入噪声
    chunk_max_size: int = 800   # 每个文档块的最大字符数
    chunk_overlap: int = 100     # 相邻块之间的重叠字符数，保持语义连贯性

    # ========== MCP（Model Context Protocol）服务配置 ==========
    # MCP 是 AI 工具调用的标准化协议，允许 AI Agent 调用外部工具获取实时信息
    # 本系统连接两个 MCP 服务：日志服务（CLS）和监控服务（Monitor）
    # CLS = Cloud Log Service，云日志服务
    mcp_cls_transport: str = "streamable-http"  # MCP 传输协议类型
    mcp_cls_url: str = "http://localhost:8003/mcp"  # CLS MCP 服务地址

    # Monitor = 云监控服务，提供 CPU、内存等指标数据
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """将分散的 MCP 配置聚合为统一的字典结构

        将 cls 和 monitor 两个 MCP 服务器的配置整合成一个字典，
        方便后续 MCP 客户端初始化时批量传入配置。

        返回值格式：
            {
                "cls": {"transport": "streamable-http", "url": "http://..."},
                "monitor": {"transport": "streamable-http", "url": "http://..."}
            }
        """
        return {
            # CLS 日志服务配置
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            # Monitor 监控服务配置
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# ========== 全局配置单例 ==========
# 在模块加载时创建唯一的 Settings 实例，整个应用共享同一份配置
# 这是单例模式的简单实现，避免重复解析 .env 文件
config = Settings()
