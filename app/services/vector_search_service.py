"""向量检索服务模块

从 Milvus 向量数据库中检索相似文档的服务。
这是 RAG 系统的查询端：
1. 将用户问题嵌入为向量
2. 在 Milvus 中搜索最相似的文档块
3. 返回检索结果供后续 LLM 生成回答使用

核心概念：
- 向量相似度：使用欧氏距离（L2）衡量向量间的相似程度
- top_k：返回最相似的 k 个结果
- ANN 检索：Approximate Nearest Neighbor，近似最近邻搜索（兼顾速度和精度）
"""

from typing import Any, Dict, List  # 类型注解

from loguru import logger  # 日志记录器
from pymilvus import Collection  # Milvus ORM Collection 对象

from app.core.milvus_client import milvus_manager  # Milvus 连接管理器
from app.services.vector_embedding_service import vector_embedding_service  # 文本嵌入服务


class SearchResult:
    """搜索结果数据类

    封装单条搜索结果，包含文档 ID、内容、相似度分数和元数据。
    提供 to_dict() 方法便于序列化为 JSON。

    字段说明：
    - id: 文档在 Milvus 中的唯一标识符
    - content: 文档的原始文本内容
    - score: 相似度分数（L2 距离，越小越相似）
    - metadata: 文档的元数据（标题、来源等）
    """

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
    ):
        """初始化搜索结果

        Args:
            id: 文档唯一标识
            content: 文档文本内容
            score: 相似度分数（L2 距离）
            metadata: 文档元数据字典
        """
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            包含所有字段的字典，可直接序列化为 JSON
        """
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """向量检索服务

    负责将用户查询转换为向量，并在 Milvus 中执行相似度搜索。

    检索流程：
    1. embed_query(): 将用户问题嵌入为向量
    2. Milvus.search(): 在向量数据库中搜索最相似的文档
    3. 解析搜索结果，返回 SearchResult 列表

    配置说明：
    - metric_type="L2": 使用欧氏距离（与索引创建时一致）
    - nprobe=10: 查询时扫描的聚类数量（影响精度和速度的平衡）
    """

    def __init__(self):
        """初始化向量检索服务"""
        logger.info("向量检索服务初始化完成")

    def search_similar_documents(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """搜索与查询文本最相似的文档

        Args:
            query: 用户的查询文本（如 "CPU 使用率过高怎么办"）
            top_k: 返回最相似的 k 个结果，默认 3 个

        Returns:
            List[SearchResult]: 搜索结果列表，按相似度从高到低排序

        Raises:
            RuntimeError: 搜索过程中发生错误时抛出

        性能说明：
        - 嵌入时间取决于文本长度，通常在 100-500ms
        - 搜索时间通常在 10-50ms（取决于数据量和索引类型）
        - 整体延迟主要来自嵌入步骤
        """
        try:
            logger.info(f"开始搜索相似文档, 查询: {query}, topK: {top_k}")

            # 步骤 1：将查询文本嵌入为向量
            # 这是最耗时的步骤（网络 API 调用）
            query_vector = vector_embedding_service.embed_query(query)
            logger.debug(f"查询向量生成成功, 维度: {len(query_vector)}")

            # 步骤 2：获取 Milvus Collection 实例
            collection: Collection = milvus_manager.get_collection()

            # 步骤 3：构建搜索参数
            # 必须与创建索引时的参数一致
            search_params = {
                "metric_type": "L2",       # 欧氏距离（与索引参数一致）
                "params": {"nprobe": 10},   # 查询时扫描的聚类数量
                                             # nprobe 越大精度越高，但速度越慢
                                             # 建议值：10-128，根据数据量调整
            }

            # 步骤 4：执行向量搜索
            # Milvus 会计算查询向量与所有文档向量的距离，返回最近的 top_k 个
            results = collection.search(
                data=[query_vector],          # 查询向量（必须是列表形式）
                anns_field="vector",          # 要搜索的向量字段名
                param=search_params,          # 搜索参数
                limit=top_k,                  # 返回的结果数量
                output_fields=["id", "content", "metadata"],  # 返回的字段
            )

            # 步骤 5：解析搜索结果
            search_results = []
            for hits in results:
                # hits 是单个查询的所有结果（这里只有 1 个查询，所以只有 1 个 hits）
                for hit in hits:
                    # hit 是单条搜索结果
                    # - hit.entity: 返回的字段值（dict 形式）
                    # - hit.distance: L2 距离分数
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,  # L2 距离，越小表示越相似
                        metadata=hit.entity.get("metadata", {}),
                    )
                    search_results.append(result)

            logger.info(f"搜索完成, 找到 {len(search_results)} 个相似文档")
            return search_results

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e


# ========== 全局向量检索服务单例 ==========
vector_search_service = VectorSearchService()
