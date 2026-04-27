"""向量嵌入服务模块

将文本转换为高维向量的服务。
向量嵌入（Text Embedding）是 RAG 系统的核心技术：
1. 文档入库时：将文本分割成 chunk，每个 chunk 生成一个向量，存入向量数据库
2. 查询时：将用户问题转换为向量，在向量数据库中检索最相似的文档块

本模块实现 LangChain 标准 Embeddings 接口，可以被 LangChain 其他组件（如 Milvus VectorStore）直接使用。

技术选型：阿里云 DashScope text-embedding-v4
- 维度：1024 维（与 Milvus collection 的 VECTOR_DIM 配置一致）
- 兼容 OpenAI Embedding API 格式
"""

from typing import List  # 类型注解：List[str] 表示字符串列表

# LangChain 标准 Embeddings 接口
# 所有嵌入实现都需要继承这个抽象基类
from langchain_core.embeddings import Embeddings

# OpenAI Python 客户端（用于调用 DashScope OpenAI 兼容 API）
from openai import OpenAI
from loguru import logger  # 项目日志

from app.config import config  # 全局配置


class DashScopeEmbeddings(Embeddings):
    """阿里云 DashScope 文本嵌入实现类

    实现 LangChain 标准 Embeddings 接口，支持：
    - embed_documents(): 批量嵌入文档列表（文档入库时使用）
    - embed_query(): 嵌入单个查询（用户查询时使用）

    DashScope text-embedding-v4 特点：
    - 输出维度：1024 维（可通过 dimensions 参数调整）
    - 支持中英文双语嵌入
    - 兼容 OpenAI Embedding API 格式

    使用示例：
        embeddings = DashScopeEmbeddings(
            api_key="your-api-key",
            model="text-embedding-v4",
            dimensions=1024
        )

        # 批量嵌入文档
        doc_vectors = embeddings.embed_documents(["文档1", "文档2"])

        # 嵌入查询
        query_vector = embeddings.embed_query("用户的问题是什么")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v4",
        dimensions: int = 1024,
    ):
        """初始化 DashScope Embeddings

        Args:
            api_key: DashScope API 密钥（必须从环境变量或 .env 文件获取）
            model: 嵌入模型名称，默认使用 text-embedding-v4
            dimensions: 向量维度，必须与 Milvus collection 配置一致（默认 1024）

        Raises:
            ValueError: API 密钥为空或为占位符时抛出
        """
        # API 密钥校验：确保已配置有效的密钥
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

        # 创建 OpenAI 客户端，配置 DashScope OpenAI 兼容端点
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model              # 嵌入模型名称
        self.dimensions = dimensions    # 向量维度

        # 日志输出：打印初始化信息（API 密钥做掩码处理）
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"DashScope Embeddings 初始化完成 - "
            f"模型: {model}, 维度: {dimensions}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """掩码 API 密钥用于日志输出

        安全原则：日志中不要暴露完整的 API 密钥。
        只显示前 8 位和后 4 位，中间用 ... 替代。

        Args:
            api_key: 原始 API 密钥

        Returns:
            str: 掩码后的密钥，如 "sk-xxxx...xxxx"
        """
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档列表（LangChain 标准接口）

        用于文档入库时，将多个文档块批量转换为向量。
        批量调用比逐个调用更高效（减少 API 往返次数）。

        Args:
            texts: 文档文本列表，每个元素是一个需要嵌入的文本块

        Returns:
            List[List[float]]: 向量列表，每个元素是对应文档的嵌入向量
                               形状为 [num_documents, dimensions]

        Raises:
            RuntimeError: API 调用失败时抛出

        性能考量：
        - 批量大小建议在 100 以内（DashScope API 限制）
        - 大量文档建议分批处理，避免超时
        """
        if not texts:
            return []

        try:
            logger.info(f"批量嵌入 {len(texts)} 个文档")

            # 调用 DashScope Embedding API
            # 使用批量接口：input 参数传入字符串列表
            response = self.client.embeddings.create(
                model=self.model,                    # 模型名称
                input=texts,                        # 文本列表（批量）
                dimensions=self.dimensions,          # 向量维度
                encoding_format="float"             # 返回 float32 格式向量
            )

            # 从响应中提取向量列表
            # response.data 是 Embedding 对象的列表，按输入顺序排列
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"批量嵌入完成, 向量维度: {len(embeddings[0])}")

            return embeddings

        except Exception as e:
            logger.error(f"批量嵌入失败: {e}")
            raise RuntimeError(f"批量嵌入失败: {e}") from e

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本（LangChain 标准接口）

        用于用户查询时，将问题文本转换为向量，以便进行相似度检索。

        Args:
            text: 查询文本，通常是用户的问题

        Returns:
            List[float]: 嵌入向量，维度为 self.dimensions

        Raises:
            ValueError: 查询文本为空时抛出
            RuntimeError: API 调用失败时抛出

        设计说明：
        - 与 embed_documents 使用相同的模型和维度
        - 确保查询向量和文档向量在同一个向量空间中
        - 只有在同一个向量空间中，才能计算相似度
        """
        if not text or not text.strip():
            raise ValueError("查询文本不能为空")

        try:
            logger.debug(f"嵌入查询, 长度: {len(text)} 字符")

            # 调用 DashScope Embedding API
            # 单个查询：input 参数传入字符串
            response = self.client.embeddings.create(
                model=self.model,                    # 模型名称
                input=text,                         # 单个查询文本
                dimensions=self.dimensions,          # 向量维度
                encoding_format="float"             # 返回 float32 格式向量
            )

            # 从响应中提取单个向量
            # response.data[0] 是第一个（也是唯一一个）Embedding 对象
            embedding = response.data[0].embedding
            logger.debug(f"查询嵌入完成, 维度: {len(embedding)}")

            return embedding

        except Exception as e:
            logger.error(f"查询嵌入失败: {e}")
            raise RuntimeError(f"查询嵌入失败: {e}") from e


# ========== 全局嵌入服务单例 ==========
# 模块加载时创建单例，整个应用共享同一个嵌入服务实例
# 配置从全局 config 读取：API 密钥、模型名称、维度
vector_embedding_service = DashScopeEmbeddings(
    api_key=config.dashscope_api_key,
    model=config.dashscope_embedding_model,
    dimensions=1024
)
