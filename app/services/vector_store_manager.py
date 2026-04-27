"""向量存储管理器模块

封装 Milvus VectorStore 的高级操作，提供：
1. 文档添加（自动向量化）
2. 文档删除（按来源文件）
3. 相似度搜索

使用 LangChain Milvus 库，提供了比直接使用 pymilvus 更简洁的 API。

设计说明：
- 与 vector_search_service 的区别：
  - vector_search_service: 直接使用 pymilvus，提供更底层的控制
  - vector_store_manager: 使用 langchain-milvus，与 LangChain 生态更好集成
- 两者可以共存，各自服务不同的使用场景
"""

from typing import List  # 类型注解

from langchain_core.documents import Document  # LangChain 文档对象
from langchain_milvus import Milvus  # LangChain Milvus VectorStore
from loguru import logger  # 日志记录器

from app.config import config  # 全局配置
from app.core.milvus_client import milvus_manager  # Milvus 连接管理器
from app.services.vector_embedding_service import vector_embedding_service  # 嵌入服务


# 统一使用的 Collection 名称
# 所有文档都存储在名为 "biz" 的 Collection 中
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器

    管理 Milvus VectorStore 的文档操作：
    - add_documents(): 添加文档（自动分割、自动向量化）
    - delete_by_source(): 删除指定文件的所有文档
    - similarity_search(): 相似度搜索

    初始化流程：
    1. 调用 milvus_manager.connect() 建立连接
    2. 创建 LangChain Milvus VectorStore 实例
    3. 配置字段映射（LangChain 字段 → Milvus 字段）

    字段映射关系：
    - LangChain Document.page_content → Milvus content 字段
    - LangChain Document.metadata → Milvus metadata 字段
    - 向量（自动生成）→ Milvus vector 字段
    - ID（用户提供或自动生成）→ Milvus id 字段
    """

    def __init__(self):
        """初始化向量存储管理器"""
        self.vector_store = None              # LangChain Milvus VectorStore 实例
        self.collection_name = COLLECTION_NAME  # Collection 名称
        self._initialize_vector_store()        # 初始化 VectorStore

    def _initialize_vector_store(self):
        """初始化 LangChain Milvus VectorStore

        必须在 PyMilvus/lc-langchain-milvus 访问 Collection 之前建立连接，
        否则会抛出 ConnectionNotExistException。

        这也是为什么要在 __init__ 中调用 milvus_manager.connect()。
        """
        try:
            # 步骤 1：确保 Milvus 连接已建立
            # milvus_manager.connect() 是幂等的，重复调用不会有问题
            # 注意：模块导入时就会执行此处，早于 FastAPI lifespan 中的 connect
            _ = milvus_manager.connect()

            # 步骤 2：配置连接参数（用于 LangChain Milvus 内部连接）
            connection_args = {
                "host": config.milvus_host,
                "port": config.milvus_port,
            }

            # 步骤 3：创建 LangChain Milvus VectorStore
            # 参数说明：
            # - embedding_function: 文本嵌入函数（自动将文本转换为向量）
            # - collection_name: Collection 名称
            # - connection_args: Milvus 连接参数
            # - auto_id=False: 使用自定义 ID（由调用方提供）
            # - drop_old=False: 不删除已存在的 Collection
            # - text_field: LangChain Document.page_content 映射到的 Milvus 字段名
            # - vector_field: 向量存储的 Milvus 字段名
            # - primary_field: 主键字段名
            # - metadata_field: 元数据字段名
            self.vector_store = Milvus(
                embedding_function=vector_embedding_service,
                collection_name=self.collection_name,
                connection_args=connection_args,
                auto_id=False,           # 使用自定义 ID
                drop_old=False,          # 不删除已存在的 Collection
                text_field="content",     # 文本内容映射到 content 字段
                vector_field="vector",    # 向量映射到 vector 字段
                primary_field="id",      # 主键映射到 id 字段
                metadata_field="metadata",  # 元数据映射到 metadata 字段
            )

            logger.info(
                f"VectorStore 初始化成功: {config.milvus_host}:{config.milvus_port}, "
                f"collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"VectorStore 初始化失败: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """批量添加文档到向量存储

        自动完成以下操作：
        1. 为每个文档生成唯一 ID
        2. 调用 embedding_function 将文本转换为向量
        3. 批量插入 Milvus

        Args:
            documents: LangChain Document 对象列表

        Returns:
            List[str]: 插入文档的 ID 列表

        Raises:
            RuntimeError: 添加失败时抛出

        性能说明：
        - LangChain Milvus 会自动批量处理，提高插入效率
        - 建议每次批量插入 100-1000 个文档
        - 文档数量过多时建议分批处理
        """
        try:
            import time  # 用于计时
            import uuid  # 用于生成唯一 ID
            start_time = time.time()

            # 为每个文档生成唯一 ID（因为 auto_id=False）
            ids = [str(uuid.uuid4()) for _ in documents]

            # LangChain Milvus 的 add_documents 会自动：
            # 1. 调用 embedding_function 生成向量
            # 2. 批量插入 Milvus
            result_ids = self.vector_store.add_documents(documents, ids=ids)

            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 VectorStore 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return result_ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """删除指定文件的所有文档

        用于重新索引文件时，先删除旧数据，避免重复。

        Args:
            file_path: 文件路径（作为 metadata["_source"] 存储的值）

        Returns:
            int: 删除的文档数量

        设计说明：
        - 文档的元数据中包含 _source 字段，记录原始文件路径
        - 使用 Milvus 的 JSON 字段查询语法删除所有属于该文件的文档
        - 如果文件从未索引过（查询失败），静默返回 0
        """
        try:
            # 获取 Milvus Collection
            collection = milvus_manager.get_collection()

            # Milvus JSON 字段查询语法：metadata["_source"] == "file_path"
            # 构建删除表达式
            expr = f'metadata["_source"] == "{file_path}"'

            # 执行删除
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0

            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count

        except Exception as e:
            # 可能是文件从未索引过，不算错误
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def get_vector_store(self) -> Milvus:
        """获取 VectorStore 实例

        用于需要直接操作 VectorStore 的场景。
        大多数情况下不需要直接调用。

        Returns:
            Milvus: LangChain Milvus VectorStore 实例
        """
        return self.vector_store

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索

        使用 LangChain Milvus VectorStore 的高级搜索接口。
        相比 vector_search_service，更加简洁，但控制粒度较粗。

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关的 LangChain Document 对象列表
        """
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# ========== 全局向量存储管理器单例 ==========
vector_store_manager = VectorStoreManager()
