"""混合检索器单元测试

测试 BM25 + 向量检索 + RRF 融合 + 重排序的完整流程。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock, patch


class TestRetrievalMetrics(unittest.TestCase):
    """检索评估指标测试"""

    def test_metrics_to_dict(self):
        """测试指标转字典"""
        from app.services.hybrid_retriever import RetrievalMetrics
        
        metrics = RetrievalMetrics(
            query="测试查询",
            top_k=5,
            bm25_results_count=5,
            bm25_avg_score=1.5,
            vector_results_count=5,
            vector_avg_score=0.8,
            fusion_results_count=5,
            recall=0.8,
            precision=0.6,
            f1_score=0.7,
            mrr=0.75,
            ndcg=0.72,
        )
        
        d = metrics.to_dict()
        
        self.assertEqual(d["query"], "测试查询")
        self.assertEqual(d["top_k"], 5)
        self.assertEqual(d["recall"], 0.8)
        self.assertEqual(d["precision"], 0.6)
        self.assertEqual(d["f1_score"], 0.7)


class TestRetrievalResult(unittest.TestCase):
    """检索结果测试"""

    def test_retrieval_result_creation(self):
        """测试检索结果创建"""
        from app.services.hybrid_retriever import RetrievalResult
        
        result = RetrievalResult(
            id="doc1",
            content="测试内容",
            score=0.95,
            bm25_score=1.2,
            vector_score=0.8,
            rank=1,
            metadata={"source": "test.txt"}
        )
        
        self.assertEqual(result.id, "doc1")
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.bm25_score, 1.2)
        self.assertEqual(result.vector_score, 0.8)
        self.assertEqual(result.rank, 1)


class TestRRFFusion(unittest.TestCase):
    """RRF 融合算法测试"""

    def test_rrf_fusion_basic(self):
        """测试基本 RRF 融合"""
        from app.services.hybrid_retriever import HybridRetriever
        from app.services.bm25_retriever import BM25Result
        
        # 创建模拟的混合检索器
        retriever = HybridRetriever(use_reranker=False)
        retriever.rrf_k = 60
        retriever.alpha_bm25 = 0.5
        retriever.alpha_vector = 0.5
        
        # BM25 结果：doc1 排第1，doc2 排第2
        bm25_results = [
            BM25Result(id="doc1", content="doc1", score=2.0, rank=1),
            BM25Result(id="doc2", content="doc2", score=1.5, rank=2),
        ]
        
        # 向量结果：doc2 排第1，doc1 排第2
        vector_results = [
            {"id": "doc2", "score": 0.9, "rank": 1},
            {"id": "doc1", "score": 0.8, "rank": 2},
        ]
        
        # 执行融合
        fused = retriever._rrf_fusion(bm25_results, vector_results, top_k=3)
        
        # doc1 和 doc2 都应该被包含
        doc_ids = [r[0] for r in fused]
        self.assertIn("doc1", doc_ids)
        self.assertIn("doc2", doc_ids)
        
        # 验证返回值格式
        if fused:
            doc_id, rrf_score, bm25_score, vector_score = fused[0]
            self.assertIsInstance(doc_id, str)
            self.assertIsInstance(rrf_score, float)
            self.assertIsInstance(bm25_score, float)
            self.assertIsInstance(vector_score, float)

    def test_rrf_with_missing_docs(self):
        """测试 RRF 融合处理缺失文档"""
        from app.services.hybrid_retriever import HybridRetriever
        from app.services.bm25_retriever import BM25Result
        
        retriever = HybridRetriever(use_reranker=False)
        retriever.rrf_k = 60
        retriever.alpha_bm25 = 0.5
        retriever.alpha_vector = 0.5
        
        # BM25 只有 doc1
        bm25_results = [
            BM25Result(id="doc1", content="doc1", score=2.0, rank=1),
        ]
        
        # 向量只有 doc2
        vector_results = [
            {"id": "doc2", "score": 0.9, "rank": 1},
        ]
        
        fused = retriever._rrf_fusion(bm25_results, vector_results, top_k=2)
        
        doc_ids = [r[0] for r in fused]
        self.assertIn("doc1", doc_ids)
        self.assertIn("doc2", doc_ids)

    def test_rrf_k_parameter(self):
        """测试 RRF k 参数影响"""
        from app.services.hybrid_retriever import HybridRetriever
        from app.services.bm25_retriever import BM25Result
        
        # 使用不同的 k 值
        for k in [10, 60, 100]:
            retriever = HybridRetriever(use_reranker=False)
            retriever.rrf_k = k
            retriever.alpha_bm25 = 0.5
            retriever.alpha_vector = 0.5
            
            bm25_results = [
                BM25Result(id="doc1", content="doc1", score=2.0, rank=1),
                BM25Result(id="doc2", content="doc2", score=1.5, rank=2),
            ]
            vector_results = [
                {"id": "doc2", "score": 0.9, "rank": 1},
                {"id": "doc1", "score": 0.8, "rank": 2},
            ]
            
            fused = retriever._rrf_fusion(bm25_results, vector_results, top_k=2)
            self.assertEqual(len(fused), 2)


class TestHybridRetrieverInit(unittest.TestCase):
    """混合检索器初始化测试"""

    def test_initialization(self):
        """测试初始化"""
        from app.services.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(
            vector_service=None,
            milvus_client=None,
            rrf_k=60,
            use_reranker=False,
            alpha_bm25=0.4,
            alpha_vector=0.6,
        )
        
        self.assertEqual(retriever.rrf_k, 60)
        self.assertEqual(retriever.alpha_bm25, 0.4)
        self.assertEqual(retriever.alpha_vector, 0.6)
        self.assertFalse(retriever.use_reranker)

    def test_get_status(self):
        """测试获取状态"""
        from app.services.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(use_reranker=False)
        status = retriever.get_status()
        
        self.assertIn("bm25", status)
        self.assertIn("rrf_k", status)
        self.assertIn("alpha_bm25", status)
        self.assertIn("alpha_vector", status)


class TestNDCGCalculation(unittest.TestCase):
    """NDCG 计算测试"""

    def test_ndcg_perfect_ranking(self):
        """测试完美排序的 NDCG"""
        from app.services.hybrid_retriever import HybridRetriever, RetrievalResult
        
        retriever = HybridRetriever(use_reranker=False)
        
        # 完美排序：前3个都是相关的
        result_ids = ["doc1", "doc2", "doc3"]
        relevant_set = {"doc1", "doc2", "doc3"}
        
        ndcg = retriever._compute_ndcg(result_ids, relevant_set)
        
        self.assertEqual(ndcg, 1.0)

    def test_ndcg_partial_relevant(self):
        """测试部分相关的 NDCG"""
        from app.services.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(use_reranker=False)
        
        # 只有第2个相关
        result_ids = ["doc1", "doc2", "doc3"]
        relevant_set = {"doc2"}
        
        ndcg = retriever._compute_ndcg(result_ids, relevant_set)
        
        # NDCG 应该小于 1.0（因为相关文档不在第一位）
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0)

    def test_ndcg_no_relevant(self):
        """测试无相关文档的 NDCG"""
        from app.services.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever(use_reranker=False)
        
        result_ids = ["doc1", "doc2", "doc3"]
        relevant_set = {"doc4", "doc5"}
        
        ndcg = retriever._compute_ndcg(result_ids, relevant_set)
        
        self.assertEqual(ndcg, 0.0)


class TestEvaluationMetrics(unittest.TestCase):
    """评估指标计算测试"""

    def test_recall_calculation(self):
        """测试召回率计算"""
        from app.services.hybrid_retriever import HybridRetriever, RetrievalMetrics, RetrievalResult
        
        retriever = HybridRetriever(use_reranker=False)
        
        metrics = RetrievalMetrics(query="test", top_k=5)
        results = [
            RetrievalResult(id="doc1", content="c1", score=0.9),
            RetrievalResult(id="doc2", content="c2", score=0.8),
            RetrievalResult(id="doc3", content="c3", score=0.7),
        ]
        relevant_ids = ["doc1", "doc2", "doc4", "doc5"]
        
        metrics = retriever._compute_evaluation_metrics(metrics, results, relevant_ids)
        
        # doc1 和 doc2 被召回，共4个相关文档
        self.assertEqual(metrics.recall, 0.5)  # 2/4

    def test_precision_calculation(self):
        """测试精确率计算"""
        from app.services.hybrid_retriever import HybridRetriever, RetrievalMetrics, RetrievalResult
        
        retriever = HybridRetriever(use_reranker=False)
        
        metrics = RetrievalMetrics(query="test", top_k=5)
        results = [
            RetrievalResult(id="doc1", content="c1", score=0.9),
            RetrievalResult(id="doc2", content="c2", score=0.8),
            RetrievalResult(id="doc3", content="c3", score=0.7),
        ]
        relevant_ids = ["doc1", "doc2", "doc4", "doc5"]
        
        metrics = retriever._compute_evaluation_metrics(metrics, results, relevant_ids)
        
        # 2个相关 / 3个检索结果
        self.assertAlmostEqual(metrics.precision, 2/3, places=2)

    def test_mrr_calculation(self):
        """测试 MRR 计算"""
        from app.services.hybrid_retriever import HybridRetriever, RetrievalMetrics, RetrievalResult
        
        retriever = HybridRetriever(use_reranker=False)
        
        metrics = RetrievalMetrics(query="test", top_k=5)
        results = [
            RetrievalResult(id="doc3", content="c3", score=0.9),  # 第1个不相关
            RetrievalResult(id="doc1", content="c1", score=0.8),  # 第2个相关
            RetrievalResult(id="doc2", content="c2", score=0.7),
        ]
        relevant_ids = ["doc1", "doc2"]
        
        metrics = retriever._compute_evaluation_metrics(metrics, results, relevant_ids)
        
        # 第1个相关文档在第2位，MRR = 1/2
        self.assertEqual(metrics.mrr, 0.5)


class TestBM25ResultsConversion(unittest.TestCase):
    """BM25 结果转换测试"""

    def test_bm25_result_creation(self):
        """测试 BM25 结果创建"""
        from app.services.bm25_retriever import BM25Result
        
        result = BM25Result(
            id="test_doc",
            content="测试内容",
            score=1.5,
            metadata={"source": "test.txt"},
            rank=1
        )
        
        self.assertEqual(result.id, "test_doc")
        self.assertEqual(result.content, "测试内容")
        self.assertEqual(result.score, 1.5)
        self.assertEqual(result.rank, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
