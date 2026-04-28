"""混合检索 + RRF 融合模块

结合 BM25 关键词检索和向量语义检索，使用 RRF 算法融合结果。

混合检索流程：
1. 并行执行 BM25 检索和向量检索
2. 使用 RRF (Reciprocal Rank Fusion) 算法融合结果
3. 可选：使用重排序模型（Reranker）进一步优化

RRF 融合算法：
RScore(d) = Σ 1 / (k + rank_i(d))
其中：
- k: 平滑参数（通常为 60）
- rank_i(d): 文档在第 i 个检索结果列表中的排名（从1开始）

使用示例：
    from app.services.hybrid_retriever import HybridRetriever
    
    retriever = HybridRetriever(vector_service, milvus_client)
    retriever.index(documents)  # 构建索引
    results = retriever.search("查询文本", top_k=10)
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import asyncio

from loguru import logger

from app.services.bm25_retriever import BM25Retriever, BM25Result


@dataclass
class RetrievalResult:
    """检索结果
    
    统一格式的检索结果，用于后续处理。
    """
    id: str
    content: str
    score: float                      # 最终融合分数
    bm25_score: float = 0.0           # BM25 原始分数
    vector_score: float = 0.0         # 向量相似度分数
    rank: int = 0                     # 最终排名
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """检索评估指标
    
    用于量化评估检索效果。
    """
    query: str
    top_k: int
    
    # BM25 指标
    bm25_results_count: int = 0
    bm25_avg_score: float = 0.0
    
    # 向量检索指标
    vector_results_count: int = 0
    vector_avg_score: float = 0.0
    
    # 融合指标
    fusion_results_count: int = 0
    rrf_time_ms: float = 0.0
    
    # 重排序指标
    reranked: bool = False
    rerank_time_ms: float = 0.0
    
    # 评估指标（如果提供了 relevant_doc_ids）
    recall: float = 0.0
    precision: float = 0.0
    f1_score: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "bm25_results_count": self.bm25_results_count,
            "bm25_avg_score": self.bm25_avg_score,
            "vector_results_count": self.vector_results_count,
            "vector_avg_score": self.vector_avg_score,
            "fusion_results_count": self.fusion_results_count,
            "rrf_time_ms": self.rrf_time_ms,
            "reranked": self.reranked,
            "rerank_time_ms": self.rerank_time_ms,
            "recall": self.recall,
            "precision": self.precision,
            "f1_score": self.f1_score,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
        }


class HybridRetriever:
    """混合检索器
    
    结合 BM25 关键词检索和向量语义检索，
    使用 RRF 算法融合结果，可选使用重排序模型优化。
    
    设计说明：
    - BM25: 擅长精确关键词匹配，适合技术术语查询
    - 向量检索: 擅长语义匹配，适合同义词、多义词查询
    - RRF: 简单有效的融合算法，无需训练参数
    - Reranker: 使用交叉编码器进行精细排序
    
    使用示例：
        retriever = HybridRetriever(
            vector_service=vector_search_service,
            milvus_client=milvus_client
        )
        retriever.index(documents)
        
        # 搜索
        results = retriever.search("查询文本")
        
        # 获取评估指标
        metrics = retriever.get_metrics()
    """
    
    def __init__(
        self,
        vector_service=None,        # VectorSearchService 实例
        milvus_client=None,         # MilvusClient 实例
        collection_name: str = "biz",
        embedding_field: str = "vector",
        text_field: str = "content",
        rrf_k: int = 60,            # RRF 平滑参数
        use_reranker: bool = True,  # 是否使用重排序
        rerank_top_n: int = 20,    # 重排序时取前 N 个结果
        alpha_bm25: float = 0.4,     # BM25 权重 (alpha + beta = 1)
        alpha_vector: float = 0.6,  # 向量检索权重
    ):
        """初始化混合检索器
        
        Args:
            vector_service: 向量搜索服务实例
            milvus_client: Milvus 客户端实例
            collection_name: 集合名称
            embedding_field: 向量字段名
            text_field: 文本字段名
            rrf_k: RRF 算法平滑参数（通常 60）
            use_reranker: 是否使用重排序模型
            rerank_top_n: 重排序时取前 N 个结果
            alpha_bm25: BM25 检索权重
            alpha_vector: 向量检索权重
        """
        self.vector_service = vector_service
        self.milvus_client = milvus_client
        self.collection_name = collection_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        
        # RRF 参数
        self.rrf_k = rrf_k
        
        # 权重参数
        self.alpha_bm25 = alpha_bm25
        self.alpha_vector = alpha_vector
        
        # 重排序配置
        self.use_reranker = use_reranker
        self.rerank_top_n = rerank_top_n
        
        # BM25 检索器
        self.bm25_retriever = BM25Retriever()
        
        # 重排序器
        self._reranker = None
        self._reranker_available = False
        
        # 检索缓存
        self._last_metrics: Optional[RetrievalMetrics] = None
        
        # 初始化重排序器
        self._init_reranker()
        
        logger.info(
            f"混合检索器初始化完成: RRF k={rrf_k}, "
            f"alpha_bm25={alpha_bm25}, alpha_vector={alpha_vector}, "
            f"use_reranker={use_reranker}"
        )
    
    def _init_reranker(self) -> None:
        """初始化重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            # 使用 BGE Reranker（推荐）
            # 支持本地模型或 HuggingFace 模型
            model_name = "BAAI/bge-reranker-large"
            
            logger.info(f"加载重排序模型: {model_name}")
            self._reranker = CrossEncoder(model_name, max_length=512)
            self._reranker_available = True
            
            logger.info("重排序模型加载成功")
            
        except ImportError:
            logger.warning(
                "sentence_transformers 未安装，"
                "将使用 RRF 分数进行排序"
            )
            self._reranker_available = False
        except Exception as e:
            logger.warning(f"重排序模型加载失败: {e}")
            self._reranker_available = False
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """构建索引
        
        为 BM25 和向量检索构建索引。
        
        Args:
            documents: 文档列表，每个文档需包含：
                - id: 文档 ID
                - content: 文本内容
                - vector: 向量（可选，不提供则自动生成）
                - metadata: 元数据（可选）
        """
        if not documents:
            logger.warning("没有文档需要索引")
            return
        
        logger.info(f"开始构建混合索引: {len(documents)} 个文档")
        
        # 1. 构建 BM25 索引
        bm25_docs = [
            {
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
            for doc in documents
        ]
        self.bm25_retriever.index(bm25_docs)
        
        # 2. 构建向量索引
        if self.milvus_client:
            try:
                self._index_vectors(documents)
            except Exception as e:
                logger.error(f"向量索引构建失败: {e}")
        
        logger.info("混合索引构建完成")
    
    def _index_vectors(self, documents: List[Dict[str, Any]]) -> None:
        """构建向量索引
        
        Args:
            documents: 文档列表
        """
        if not self.vector_service:
            logger.warning("向量服务未配置，跳过向量索引")
            return
        
        # 生成向量的文档数据
        docs_for_embedding = []
        
        for doc in documents:
            # 检查是否已有向量
            if "vector" in doc and doc["vector"]:
                # 使用提供的向量
                continue
            
            # 需要生成向量
            docs_for_embedding.append(doc)
        
        if not docs_for_embedding:
            logger.info("所有文档已有向量，跳过向量生成")
            return
        
        logger.info(f"为 {len(docs_for_embedding)} 个文档生成向量")
        
        # 批量生成向量
        texts = [doc["content"] for doc in docs_for_embedding]
        
        try:
            # 使用向量服务生成嵌入
            embeddings = self.vector_service.embed_texts(texts)
            
            # 更新文档向量
            for i, doc in enumerate(docs_for_embedding):
                doc["vector"] = embeddings[i]
            
        except Exception as e:
            logger.error(f"向量生成失败: {e}")
            raise
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        relevant_doc_ids: Optional[List[str]] = None,  # 用于评估
        return_metrics: bool = True,
    ) -> List[RetrievalResult]:
        """执行混合检索
        
        Args:
            query: 查询文本
            top_k: 返回的最多结果数
            filters: Milvus 过滤条件
            relevant_doc_ids: 相关文档 ID 列表（用于评估）
            return_metrics: 是否返回评估指标
            
        Returns:
            List[RetrievalResult]: 排序后的检索结果
        """
        import time
        
        start_time = time.time()
        
        if not query:
            return []
        
        logger.debug(f"执行混合检索: {query[:50]}...")
        
        # 1. 执行 BM25 检索
        bm25_start = time.time()
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
        bm25_time = (time.time() - bm25_start) * 1000
        
        # 2. 执行向量检索
        vector_start = time.time()
        vector_results = self._search_vectors(query, top_k * 2, filters)
        vector_time = (time.time() - vector_start) * 1000
        
        # 3. RRF 融合
        rrf_start = time.time()
        fused_results = self._rrf_fusion(bm25_results, vector_results, top_k)
        rrf_time = (time.time() - rrf_start) * 1000
        
        # 4. 重排序（可选）
        rerank_time = 0.0
        if self.use_reranker and self._reranker_available and len(fused_results) > 0:
            rerank_start = time.time()
            fused_results = self._rerank_results(query, fused_results)
            rerank_time = (time.time() - rerank_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # 5. 构建结果
        results = []
        for rank, (doc_id, rrf_score, bm25_score, vector_score) in enumerate(fused_results[:top_k]):
            result = RetrievalResult(
                id=doc_id,
                content=self._get_content(doc_id),
                score=rrf_score,
                bm25_score=bm25_score,
                vector_score=vector_score,
                rank=rank + 1,
                metadata=self._get_metadata(doc_id)
            )
            results.append(result)
        
        # 6. 计算评估指标
        if return_metrics:
            self._last_metrics = self._compute_metrics(
                query=query,
                top_k=top_k,
                bm25_results=bm25_results,
                vector_results=vector_results,
                results=results,
                bm25_time=bm25_time,
                vector_time=vector_time,
                rrf_time=rrf_time,
                rerank_time=rerank_time,
                relevant_doc_ids=relevant_doc_ids
            )
        
        logger.info(
            f"混合检索完成: {len(results)} 个结果, "
            f"BM25: {bm25_time:.1f}ms, "
            f"向量: {vector_time:.1f}ms, "
            f"RRF: {rrf_time:.1f}ms, "
            f"总计: {total_time:.1f}ms"
        )
        
        return results
    
    def _search_vectors(
        self, 
        query: str, 
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """执行向量检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            filters: 过滤条件
            
        Returns:
            List[Dict]: 向量检索结果
        """
        if not self.vector_service:
            return []
        
        try:
            # 生成查询向量
            query_vector = self.vector_service.embed_texts([query])[0]
            
            # 执行 Milvus 搜索
            results = self.milvus_client.collection.search(
                data=[query_vector],
                anns_field=self.embedding_field,
                param={"metric_type": "COSINE", "params": {}},
                limit=top_k,
                output_fields=["id", self.text_field, "metadata"]
            )
            
            # 解析结果
            parsed_results = []
            if results and len(results) > 0:
                for hits in results:
                    for hit in hits:
                        parsed_results.append({
                            "id": str(hit.entity.get("id", "")),
                            "score": hit.distance,
                            "content": hit.entity.get(self.text_field, ""),
                            "metadata": hit.entity.get("metadata", {})
                        })
            
            return parsed_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []
    
    def _rrf_fusion(
        self,
        bm25_results: List[BM25Result],
        vector_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Tuple[str, float, float, float]]:
        """RRF 融合
        
        使用 Reciprocal Rank Fusion 算法融合多个检索结果。
        
        公式：
        RScore(d) = Σ 1 / (k + rank_i(d))
        
        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            top_k: 返回数量
            
        Returns:
            List[Tuple]: (doc_id, rrf_score, bm25_score, vector_score)
        """
        # 构建排名字典
        bm25_ranks: Dict[str, int] = {
            r.id: r.rank for r in bm25_results
        }
        bm25_scores: Dict[str, float] = {
            r.id: r.score for r in bm25_results
        }
        
        vector_ranks: Dict[str, int] = {}
        vector_scores: Dict[str, float] = {}
        for i, r in enumerate(vector_results):
            vector_ranks[r["id"]] = r.get("rank", i + 1)
            vector_scores[r["id"]] = r.get("score", 0.0)
        
        # 获取所有文档 ID
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # 计算 RRF 分数
        rrf_scores: Dict[str, float] = {}
        
        for doc_id in all_doc_ids:
            score = 0.0
            
            # BM25 的 RRF 分数
            if doc_id in bm25_ranks:
                rank = bm25_ranks[doc_id]
                score += self.alpha_bm25 / (self.rrf_k + rank)
            
            # 向量检索的 RRF 分数
            if doc_id in vector_ranks:
                rank = vector_ranks[doc_id]
                score += self.alpha_vector / (self.rrf_k + rank)
            
            rrf_scores[doc_id] = score
        
        # 按 RRF 分数排序
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 构建结果列表
        results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vec_score = vector_scores.get(doc_id, 0.0)
            results.append((doc_id, rrf_score, bm25_score, vec_score))
        
        return results
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """使用交叉编码器重排序结果
        
        Args:
            query: 查询文本
            results: 待重排序的结果
            
        Returns:
            List[RetrievalResult]: 重排序后的结果
        """
        if not self._reranker_available or not results:
            return results
        
        try:
            # 准备输入对
            pairs = [(query, result.content) for result in results]
            
            # 获取重排序分数
            scores = self._reranker.predict(pairs)
            
            # 按新分数排序
            for i, result in enumerate(results):
                result.score = float(scores[i])
            
            results.sort(key=lambda x: x.score, reverse=True)
            
            # 更新排名
            for i, result in enumerate(results):
                result.rank = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            return results
    
    def _get_content(self, doc_id: str) -> str:
        """获取文档内容"""
        # 尝试从 BM25 检索器获取
        if doc_id in self.bm25_retriever.documents:
            return self.bm25_retriever.documents[doc_id].content
        
        # 尝试从向量检索结果获取
        # （需要在搜索时缓存结果）
        return ""
    
    def _get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """获取文档元数据"""
        if doc_id in self.bm25_retriever.documents:
            return self.bm25_retriever.documents[doc_id].metadata
        return {}
    
    def _compute_metrics(
        self,
        query: str,
        top_k: int,
        bm25_results: List[BM25Result],
        vector_results: List[Dict[str, Any]],
        results: List[RetrievalResult],
        bm25_time: float,
        vector_time: float,
        rrf_time: float,
        rerank_time: float,
        relevant_doc_ids: Optional[List[str]] = None
    ) -> RetrievalMetrics:
        """计算检索评估指标
        
        Args:
            query: 查询文本
            top_k: top_k 参数
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果
            results: 最终结果
            bm25_time: BM25 检索耗时
            vector_time: 向量检索耗时
            rrf_time: RRF 融合耗时
            rerank_time: 重排序耗时
            relevant_doc_ids: 相关文档 ID（用于计算召回率等）
            
        Returns:
            RetrievalMetrics: 评估指标
        """
        metrics = RetrievalMetrics(
            query=query,
            top_k=top_k,
            bm25_results_count=len(bm25_results),
            vector_results_count=len(vector_results),
            fusion_results_count=len(results),
            rrf_time_ms=rrf_time + bm25_time + vector_time,
            reranked=self.use_reranker and self._reranker_available,
            rerank_time_ms=rerank_time
        )
        
        # BM25 平均分数
        if bm25_results:
            metrics.bm25_avg_score = sum(r.score for r in bm25_results) / len(bm25_results)
        
        # 向量检索平均分数
        if vector_results:
            metrics.vector_avg_score = sum(r.get("score", 0) for r in vector_results) / len(vector_results)
        
        # 如果提供了相关文档，计算评估指标
        if relevant_doc_ids:
            metrics = self._compute_evaluation_metrics(metrics, results, relevant_doc_ids)
        
        return metrics
    
    def _compute_evaluation_metrics(
        self,
        metrics: RetrievalMetrics,
        results: List[RetrievalResult],
        relevant_doc_ids: List[str]
    ) -> RetrievalMetrics:
        """计算评估指标（召回率、精确率、F1、MRR、NDCG）
        
        Args:
            metrics: 基础指标
            results: 检索结果
            relevant_doc_ids: 相关文档 ID 集合
            
        Returns:
            RetrievalMetrics: 包含评估指标的指标对象
        """
        if not relevant_doc_ids:
            return metrics
        
        relevant_set = set(relevant_doc_ids)
        result_ids = [r.id for r in results]
        result_set = set(result_ids)
        
        # 计算交集
        retrieved_relevant = relevant_set & result_set
        
        # 召回率
        metrics.recall = len(retrieved_relevant) / len(relevant_set) if relevant_set else 0.0
        
        # 精确率
        metrics.precision = len(retrieved_relevant) / len(result_set) if result_set else 0.0
        
        # F1 分数
        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = (
                2 * metrics.precision * metrics.recall / 
                (metrics.precision + metrics.recall)
            )
        
        # MRR (Mean Reciprocal Rank)
        mrr_score = 0.0
        for i, doc_id in enumerate(result_ids):
            if doc_id in relevant_set:
                mrr_score = 1.0 / (i + 1)
                break
        metrics.mrr = mrr_score
        
        # NDCG (Normalized Discounted Cumulative Gain)
        metrics.ndcg = self._compute_ndcg(result_ids, relevant_set)
        
        logger.info(
            f"检索评估: recall={metrics.recall:.4f}, "
            f"precision={metrics.precision:.4f}, "
            f"f1={metrics.f1_score:.4f}, "
            f"mrr={metrics.mrr:.4f}, "
            f"ndcg={metrics.ndcg:.4f}"
        )
        
        return metrics
    
    def _compute_ndcg(
        self,
        result_ids: List[str],
        relevant_set: set
    ) -> float:
        """计算 NDCG
        
        Args:
            result_ids: 检索结果 ID 列表
            relevant_set: 相关文档 ID 集合
            
        Returns:
            float: NDCG 值
        """
        def dcg(gains: List[float]) -> float:
            return sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        
        # 计算 DCG
        gains = [1.0 if doc_id in relevant_set else 0.0 for doc_id in result_ids]
        dcg_value = dcg(gains)
        
        # 计算 IDCG（理想情况下的 DCG）
        ideal_gains = [1.0] * len(relevant_set)
        idcg_value = dcg(ideal_gains)
        
        # 返回 NDCG
        if idcg_value == 0:
            return 0.0
        return dcg_value / idcg_value
    
    def get_metrics(self) -> Optional[RetrievalMetrics]:
        """获取上次检索的评估指标
        
        Returns:
            Optional[RetrievalMetrics]: 上次的评估指标
        """
        return self._last_metrics
    
    def clear_metrics(self) -> None:
        """清除评估指标"""
        self._last_metrics = None
    
    def get_status(self) -> Dict[str, Any]:
        """获取检索器状态
        
        Returns:
            Dict: 状态信息
        """
        return {
            "bm25": self.bm25_retriever.get_stats(),
            "reranker_available": self._reranker_available,
            "reranker_model": "BAAI/bge-reranker-large" if self._reranker_available else None,
            "rrf_k": self.rrf_k,
            "alpha_bm25": self.alpha_bm25,
            "alpha_vector": self.alpha_vector,
        }
