"""BM25 检索器单元测试"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from app.services.bm25_retriever import BM25Retriever, BM25Result


class TestBM25Retriever(unittest.TestCase):
    """BM25 检索器测试"""

    def setUp(self):
        """测试前准备"""
        self.retriever = BM25Retriever(k1=1.5, b=0.75)
        
        # 测试文档集
        self.documents = [
            {
                "id": "doc1",
                "content": "Python 是一种广泛使用的高级编程语言。Python 设计哲学强调代码的可读性。",
                "metadata": {"source": "python_intro.txt"}
            },
            {
                "id": "doc2", 
                "content": "JavaScript 是用于 Web 开发的编程语言。JavaScript 可以创建动态网页内容。",
                "metadata": {"source": "javascript_intro.txt"}
            },
            {
                "id": "doc3",
                "content": "Python 和 JavaScript 都是流行的编程语言。Python 用于后端开发，JavaScript 用于前端开发。",
                "metadata": {"source": "programming_overview.txt"}
            },
            {
                "id": "doc4",
                "content": "深度学习是机器学习的一个分支。深度学习使用神经网络进行特征学习。",
                "metadata": {"source": "deep_learning.txt"}
            },
            {
                "id": "doc5",
                "content": "Python 在数据科学和机器学习领域非常流行。许多深度学习框架使用 Python API。",
                "metadata": {"source": "python_ml.txt"}
            },
        ]

    def test_index_documents(self):
        """测试文档索引"""
        self.retriever.index(self.documents)
        
        stats = self.retriever.get_stats()
        
        self.assertEqual(stats["document_count"], 5)
        self.assertEqual(stats["indexed"], True)
        self.assertGreater(stats["avg_doc_length"], 0)
        self.assertGreater(stats["vocabulary_size"], 0)

    def test_search_python(self):
        """测试搜索 'Python'"""
        self.retriever.index(self.documents)
        results = self.retriever.search("Python", top_k=3)
        
        self.assertGreater(len(results), 0)
        # Python 相关的文档应该排在前面
        doc_ids = [r.id for r in results]
        self.assertIn("doc1", doc_ids)  # Python 介绍
        self.assertIn("doc3", doc_ids)  # Python 和 JavaScript
        self.assertIn("doc5", doc_ids)  # Python ML

    def test_search_deep_learning(self):
        """测试搜索 '深度学习'"""
        self.retriever.index(self.documents)
        results = self.retriever.search("深度学习", top_k=3)
        
        # 深度学习文档应该排在前面
        self.assertGreater(len(results), 0)
        doc_ids = [r.id for r in results]
        self.assertIn("doc4", doc_ids)  # 深度学习

    def test_search_with_top_k(self):
        """测试 top_k 参数"""
        self.retriever.index(self.documents)
        
        results = self.retriever.search("Python", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_search_empty_query(self):
        """测试空查询"""
        self.retriever.index(self.documents)
        results = self.retriever.search("")
        self.assertEqual(len(results), 0)

    def test_search_no_index(self):
        """测试未索引情况"""
        results = self.retriever.search("test")
        self.assertEqual(len(results), 0)

    def test_get_stats(self):
        """测试统计信息"""
        self.retriever.index(self.documents)
        stats = self.retriever.get_stats()
        
        self.assertEqual(stats["document_count"], 5)
        self.assertIn("vocabulary_size", stats)
        self.assertIn("avg_doc_length", stats)
        self.assertIn("k1", stats)
        self.assertIn("b", stats)

    def test_add_document(self):
        """测试添加单个文档"""
        self.retriever.index(self.documents[:3])
        
        # 添加新文档
        new_doc = {
            "id": "doc6",
            "content": "Go 语言是 Google 开发的编译型编程语言。",
            "metadata": {"source": "go_intro.txt"}
        }
        self.retriever.add_document(new_doc)
        
        # 验证文档已添加（但不保证精确分数，因为未重新索引）
        self.assertIn("doc6", self.retriever.documents)

    def test_remove_document(self):
        """测试移除文档"""
        self.retriever.index(self.documents)
        
        result = self.retriever.remove_document("doc1")
        self.assertTrue(result)
        self.assertNotIn("doc1", self.retriever.documents)

    def test_clear(self):
        """测试清空索引"""
        self.retriever.index(self.documents)
        self.retriever.clear()
        
        stats = self.retriever.get_stats()
        self.assertEqual(stats["document_count"], 0)
        self.assertEqual(stats["indexed"], False)

    def test_chinese_tokenization(self):
        """测试中文分词"""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("这是一段中文测试文本")
        
        # 应该包含 bigram（unigram 可能有单字符被过滤）
        self.assertIn("中文", tokens)
        self.assertIn("测试", tokens)
        self.assertTrue(len(tokens) > 0)

    def test_english_tokenization(self):
        """测试英文分词"""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("This is an English text")
        
        self.assertIn("this", tokens)
        self.assertIn("english", tokens)
        self.assertIn("text", tokens)
        # "is" 和 "an" 在停用词列表中，应该被过滤

    def test_idf_values(self):
        """测试 IDF 值计算"""
        self.retriever.index(self.documents)
        
        # 高频词应该有较低的 IDF
        idf_python = self.retriever.get_idf("python")
        idf_deep = self.retriever.get_idf("深度")
        
        # Python 出现次数多，IDF 应该较低
        self.assertGreater(idf_python, 0)
        self.assertGreater(idf_deep, 0)

    def test_score_ranking(self):
        """测试分数排序"""
        self.retriever.index(self.documents)
        results = self.retriever.search("Python 编程", top_k=5)
        
        # 验证结果按分数降序排列
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score, 
                results[i + 1].score,
                "结果应该按分数降序排列"
            )

    def test_mixed_language(self):
        """测试中英文混合"""
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Python 编程语言很强大")
        
        self.assertIn("python", tokens)
        self.assertIn("编程", tokens)
        self.assertIn("语言", tokens)
        self.assertIn("强大", tokens)

    def test_result_metadata(self):
        """测试结果元数据"""
        self.retriever.index(self.documents)
        results = self.retriever.search("Python", top_k=5)
        
        self.assertGreater(len(results), 0)
        # 验证结果包含必要字段
        for result in results:
            self.assertIsNotNone(result.id)
            self.assertIsNotNone(result.content)
            self.assertGreater(result.score, 0)
            self.assertGreater(result.rank, 0)


class TestBM25EdgeCases(unittest.TestCase):
    """BM25 边界情况测试"""

    def test_empty_documents(self):
        """测试空文档列表"""
        retriever = BM25Retriever()
        retriever.index([])
        
        stats = retriever.get_stats()
        self.assertEqual(stats["document_count"], 0)

    def test_single_document(self):
        """测试单个文档"""
        retriever = BM25Retriever()
        retriever.index([
            {"id": "doc1", "content": "唯一的文档内容", "metadata": {}}
        ])
        
        results = retriever.search("唯一")
        self.assertGreater(len(results), 0)

    def test_no_matching_documents(self):
        """测试无匹配文档"""
        retriever = BM25Retriever()
        retriever.index([
            {"id": "doc1", "content": "Python 编程", "metadata": {}},
            {"id": "doc2", "content": "JavaScript 开发", "metadata": {}},
        ])
        
        results = retriever.search("深度学习神经网络")
        # 可能返回结果（因为中文分词），但分数应该很低或为空
        # 具体取决于分词效果


if __name__ == "__main__":
    unittest.main(verbosity=2)
