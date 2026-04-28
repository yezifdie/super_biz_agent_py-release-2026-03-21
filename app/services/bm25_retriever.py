"""BM25 关键词检索模块

实现 BM25 (Best Matching 25) 检索算法，用于关键词匹配。

BM25 特性：
- 基于词频和逆文档频率的统计检索
- 对长文档有更好的规范化处理
- 对查询词项进行 IDF 加权

与向量检索的对比：
- BM25: 擅长精确关键词匹配，适合技术术语查询
- 向量检索: 擅长语义相似度匹配，适合同义词查询
- 混合检索: 结合两者优势，提升召回率和精确率

使用示例：
    from app.services.hybrid_retriever import BM25Retriever
    
    retriever = BM25Retriever()
    retriever.index(documents)
    results = retriever.search("查询文本", top_k=10)
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from loguru import logger


@dataclass
class BM25Document:
    """BM25 索引文档
    
    存储文档内容和预处理后的词项列表。
    """
    id: str                           # 文档 ID
    content: str                      # 原始内容
    tokens: List[str]                 # 分词后的词项列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据


@dataclass 
class BM25Result:
    """BM25 检索结果
    
    包含文档 ID、原始内容、BM25 分数和元数据。
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0


class BM25Retriever:
    """BM25 关键词检索器
    
    实现经典的 BM25 检索算法。
    
    参数说明：
    - k1: 词频饱和参数 (default: 1.5)
      控制词频增长对分数的影响
      较低值使长文档不那么占优
    - b: 文档长度规范化参数 (default: 0.75)
      控制文档长度对分数的影响
      较高值对短文档更友好
    
    公式：
    BM25(d, q) = Σ IDF(qi) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d|/avgdl))
    IDF(qi) = log((N - n + 0.5) / (n + 0.5) + 1)
    """
    
    def __init__(
        self, 
        k1: float = 1.5,
        b: float = 0.75,
        min_token_length: int = 2,
        stop_words: Optional[set] = None
    ):
        """初始化 BM25 检索器
        
        Args:
            k1: 词频饱和参数，控制词频的影响力
            b: 文档长度规范化参数
            min_token_length: 最小词项长度，短于此长度的词项会被过滤
            stop_words: 停用词集合
        """
        self.k1 = k1
        self.b = b
        self.min_token_length = min_token_length
        
        # 默认中文停用词（简化版）
        self.stop_words = stop_words or {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '个',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        }
        
        # 索引数据
        self.documents: Dict[str, BM25Document] = {}
        self.avg_doc_length: float = 0.0
        self.N: int = 0  # 文档总数
        
        # 预计算的 IDF 值
        self.idf: Dict[str, float] = {}
        
        # 文档频率
        self.doc_freq: Dict[str, int] = defaultdict(int)
        
        # 是否已索引
        self._indexed = False
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """构建 BM25 索引
        
        Args:
            documents: 文档列表，每个文档需包含 'id' 和 'content' 字段
        """
        if not documents:
            logger.warning("没有文档需要索引")
            return
        
        logger.info(f"开始构建 BM25 索引: {len(documents)} 个文档")
        
        # 清空旧索引
        self.documents.clear()
        self.doc_freq.clear()
        self.idf.clear()
        self._indexed = False
        
        # 解析文档
        for doc in documents:
            doc_id = str(doc.get('id', ''))
            content = str(doc.get('content', ''))
            metadata = doc.get('metadata', {})
            
            if not doc_id or not content:
                logger.warning(f"跳过无效文档: {doc_id}")
                continue
            
            # 分词
            tokens = self._tokenize(content)
            
            # 创建文档对象
            bm25_doc = BM25Document(
                id=doc_id,
                content=content,
                tokens=tokens,
                metadata=metadata
            )
            self.documents[doc_id] = bm25_doc
            
            # 统计文档频率（只统计去重后的词项）
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freq[token] += 1
        
        self.N = len(self.documents)
        
        # 计算平均文档长度
        total_length = sum(len(doc.tokens) for doc in self.documents.values())
        self.avg_doc_length = total_length / self.N if self.N > 0 else 0
        
        # 计算 IDF 值
        self._compute_idf()
        
        self._indexed = True
        
        logger.info(
            f"BM25 索引构建完成: {self.N} 个文档, "
            f"平均长度: {self.avg_doc_length:.1f}, "
            f"词表大小: {len(self.doc_freq)}"
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词
        
        简单的中英文混合分词。
        中文按字符 n-gram 处理，英文按单词处理。
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 词项列表
        """
        if not text:
            return []
        
        tokens = []
        
        # 预处理：转小写，移除特殊字符
        text = text.lower()
        
        # 提取中文（连续汉字序列）
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_parts = chinese_pattern.findall(text)
        
        # 提取英文和数字
        english_pattern = re.compile(r'[a-z0-9]+')
        english_parts = english_pattern.findall(text)
        
        # 处理中文字符（按字符 unigram + bigram）
        for chinese_text in chinese_parts:
            chars = list(chinese_text)
            # Unigram
            tokens.extend(chars)
            # Bigram（窗口为2的滑动）
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        
        # 处理英文（按单词）
        for word in english_parts:
            if len(word) >= self.min_token_length:
                if word not in self.stop_words:
                    tokens.append(word)
        
        # 过滤短词项
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        return tokens
    
    def _compute_idf(self) -> None:
        """计算所有词项的 IDF 值"""
        for token, df in self.doc_freq.items():
            # 使用平滑的 IDF 公式
            # IDF = log((N - n + 0.5) / (n + 0.5) + 1)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            self.idf[token] = max(idf, 0)  # 确保 IDF 非负
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        min_score: Optional[float] = None
    ) -> List[BM25Result]:
        """搜索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回的最多结果数
            min_score: 最小分数阈值，只返回分数 >= 此值的文档
            
        Returns:
            List[BM25Result]: 排序后的检索结果列表
        """
        if not self._indexed:
            logger.warning("BM25 索引尚未构建，调用 index() 方法")
            return []
        
        if not query:
            return []
        
        # 对查询进行分词
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # 计算每个文档的 BM25 分数
        scores: Dict[str, float] = {}
        
        for doc_id, doc in self.documents.items():
            score = self._compute_bm25_score(doc, query_tokens)
            if score > 0:
                if min_score is None or score >= min_score:
                    scores[doc_id] = score
        
        # 按分数排序
        sorted_results = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 构建结果列表
        results = []
        for rank, (doc_id, score) in enumerate(sorted_results[:top_k]):
            doc = self.documents[doc_id]
            results.append(BM25Result(
                id=doc_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                rank=rank + 1
            ))
        
        return results
    
    def _compute_bm25_score(
        self, 
        doc: BM25Document, 
        query_tokens: List[str]
    ) -> float:
        """计算单个文档的 BM25 分数
        
        Args:
            doc: 目标文档
            query_tokens: 查询词项列表
            
        Returns:
            float: BM25 分数
        """
        # 词频统计
        doc_tf = Counter(doc.tokens)
        doc_length = len(doc.tokens)
        
        score = 0.0
        
        for token in query_tokens:
            if token not in self.idf:
                continue
            
            tf = doc_tf.get(token, 0)
            idf = self.idf[token]
            
            if tf > 0:
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                score += idf * (numerator / denominator)
        
        return score
    
    def get_term_freq(self, doc_id: str, term: str) -> int:
        """获取词项在文档中的频率
        
        Args:
            doc_id: 文档 ID
            term: 词项
            
        Returns:
            int: 词频
        """
        if doc_id not in self.documents:
            return 0
        
        doc = self.documents[doc_id]
        return doc.tokens.count(term.lower())
    
    def get_document_freq(self, term: str) -> int:
        """获取词项的文档频率
        
        Args:
            term: 词项
            
        Returns:
            int: 包含该词项的文档数
        """
        return self.doc_freq.get(term.lower(), 0)
    
    def get_idf(self, term: str) -> float:
        """获取词项的 IDF 值
        
        Args:
            term: 词项
            
        Returns:
            float: IDF 值
        """
        return self.idf.get(term.lower(), 0.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "document_count": self.N,
            "vocabulary_size": len(self.doc_freq),
            "avg_doc_length": self.avg_doc_length,
            "indexed": self._indexed,
            "k1": self.k1,
            "b": self.b,
        }
    
    def add_document(self, doc: Dict[str, Any]) -> None:
        """添加单个文档到索引
        
        注意：调用此方法后，之前计算的分数可能不再准确，
        如需精确分数，请重新调用 index()。
        
        Args:
            doc: 文档数据，包含 'id' 和 'content'
        """
        doc_id = str(doc.get('id', ''))
        content = str(doc.get('content', ''))
        metadata = doc.get('metadata', {})
        
        if not doc_id or not content:
            return
        
        tokens = self._tokenize(content)
        
        bm25_doc = BM25Document(
            id=doc_id,
            content=content,
            tokens=tokens,
            metadata=metadata
        )
        self.documents[doc_id] = bm25_doc
        self._indexed = False  # 标记需要重新索引
    
    def remove_document(self, doc_id: str) -> bool:
        """从索引中移除文档
        
        Args:
            doc_id: 文档 ID
            
        Returns:
            bool: 是否成功移除
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            self._indexed = False
            return True
        return False
    
    def clear(self) -> None:
        """清空所有索引数据"""
        self.documents.clear()
        self.doc_freq.clear()
        self.idf.clear()
        self.avg_doc_length = 0.0
        self.N = 0
        self._indexed = False
        logger.info("BM25 索引已清空")
