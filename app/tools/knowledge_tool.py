"""知识检索工具模块

从向量数据库（Milvus）中检索与用户问题相关的文档片段。
这是 RAG（检索增强生成）系统的核心工具。

增强功能（v2）：
1. 混合检索：结合 BM25 关键词检索和向量语义检索
2. RRF 融合：使用 Reciprocal Rank Fusion 算法合并结果
3. 重排序：可选使用 BGE-Reranker 模型精细排序
4. 评估指标：计算召回率、精确率、MRR、NDCG 等

工作原理：
1. 用户提出问题
2. 并行执行 BM25 检索和向量检索
3. 使用 RRF 算法融合两个检索结果
4. 可选：使用重排序模型进一步优化
5. 返回格式化的文档内容作为上下文

技术细节：
- BM25: 擅长精确关键词匹配，适合技术术语查询
- 向量检索: 擅长语义匹配，适合同义词、多义词查询
- RRF: 简单有效的融合算法，无需训练参数
- Reranker: 使用交叉编码器进行精细排序

@tool 装饰器参数：
- response_format="content_and_artifact": 返回值包含内容和原始对象两部分
  用于需要同时返回格式化文本和原始文档的场景
"""

from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager

# 尝试导入混合检索器
try:
    from app.services.hybrid_retriever import HybridRetriever, RetrievalResult
    _hybrid_retriever_available = True
except ImportError as e:
    _hybrid_retriever_available = False
    logger.warning(f"混合检索器不可用: {e}")


def _get_hybrid_retriever() -> Optional['HybridRetriever']:
    """获取或创建混合检索器实例（单例模式）

    Returns:
        HybridRetriever 实例或 None（如果不可用）
    """
    if not _hybrid_retriever_available:
        return None

    global _hybrid_retriever
    if '_hybrid_retriever' not in globals():
        from app.services.vector_search_service import vector_search_service
        from app.core.milvus_client import milvus_client

        _hybrid_retriever = HybridRetriever(
            vector_service=vector_search_service,
            milvus_client=milvus_client,
            collection_name="biz",
            use_reranker=True,
            rrf_k=60,
            alpha_bm25=0.4,
            alpha_vector=0.6,
        )

    return _hybrid_retriever


@tool(response_format="content_and_artifact")
def retrieve_knowledge(
    query: str,
    top_k: Optional[int] = None,
    use_hybrid: bool = True,
    relevant_doc_ids: Optional[List[str]] = None,
) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，Agent 会自动调用此工具。

    调用场景示例：
    - 用户问："CPU 使用率过高怎么处理？"
    - Agent 分析：需要查询运维知识库中的相关文档
    - 调用 retrieve_knowledge("CPU 使用率过高")
    - 返回相关文档片段
    - Agent 基于文档内容生成回答

    工作流程：
    1. 尝试使用混合检索（BM25 + 向量 + RRF）
    2. 如果混合检索不可用，降级到向量检索
    3. 格式化文档为字符串上下文
    4. 返回 (上下文文本, 原始文档列表)

    Args:
        query: 用户的查询文本
            通常是用户问题的原文或关键词
            示例："CPU 使用率过高怎么办"
        top_k: 返回的文档数量（默认使用 config.rag_top_k）
        use_hybrid: 是否使用混合检索（默认 True）
        relevant_doc_ids: 相关文档 ID 列表（用于评估，可选）

    Returns:
        Tuple[str, List[Document]]: 元组包含
        - 第一个元素：格式化的上下文文本（字符串）
        - 第二个元素：原始 Document 对象列表

        示例返回值：
        (
            "【参考资料 1】\n标题: CPU 过高处理\n来源: cpu_high.md\n内容:\n当 CPU 使用率超过 80% 时...\n",
            [Document(...), Document(...)]
        )

        无结果时：
        ("没有找到相关信息。", [])
    """
    try:
        k = top_k if top_k is not None else config.rag_top_k
        logger.info(f"知识检索工具被调用: query='{query}', top_k={k}, use_hybrid={use_hybrid}")

        docs: List[Document] = []
        retrieval_info: Dict[str, Any] = {}

        # 1. 尝试使用混合检索
        if use_hybrid and _hybrid_retriever_available and config.use_hybrid_retrieval:
            try:
                retriever = _get_hybrid_retriever()
                if retriever:
                    results = retriever.search(
                        query=query,
                        top_k=k,
                        relevant_doc_ids=relevant_doc_ids,
                    )

                    # 转换为 Document 对象
                    docs = _results_to_documents(results)

                    # 获取评估指标
                    metrics = retriever.get_metrics()
                    if metrics:
                        retrieval_info = metrics.to_dict()

                    logger.info(
                        f"混合检索完成: {len(docs)} 个结果, "
                        f"recall={retrieval_info.get('recall', 0):.4f}"
                    )
                else:
                    raise RuntimeError("混合检索器初始化失败")
            except Exception as e:
                logger.warning(f"混合检索失败: {e}，降级到向量检索")
                docs = _fallback_vector_search(query, k)
        else:
            # 2. 使用传统向量检索
            docs = _fallback_vector_search(query, k)

        # 无结果处理
        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 3. 格式化文档为上下文字符串
        context = format_docs(docs)

        # 添加检索信息（用于调试和分析）
        if retrieval_info:
            logger.info(f"检索指标: {retrieval_info}")

        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        # 返回错误信息，不抛出异常（让 Agent 能继续处理）
        return f"检索知识时发生错误: {str(e)}", []


def _fallback_vector_search(query: str, k: int) -> List[Document]:
    """降级到传统向量检索

    Args:
        query: 查询文本
        k: 返回数量

    Returns:
        List[Document]: 文档列表
    """
    logger.info("使用降级向量检索")

    try:
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        return retriever.invoke(query)
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        return []


def _results_to_documents(results: List['RetrievalResult']) -> List[Document]:
    """将混合检索结果转换为 Document 对象

    Args:
        results: 混合检索结果

    Returns:
        List[Document]: LangChain Document 对象列表
    """
    docs = []

    for result in results:
        # 构建 Document 对象
        metadata = result.metadata.copy() if result.metadata else {}
        metadata["_score"] = result.score
        metadata["_bm25_score"] = result.bm25_score
        metadata["_vector_score"] = result.vector_score
        metadata["_rank"] = result.rank

        doc = Document(
            page_content=result.content,
            metadata=metadata
        )
        docs.append(doc)

    return docs


def get_retrieval_metrics() -> Optional[Dict[str, Any]]:
    """获取上次检索的评估指标

    Returns:
        Dict: 评估指标或 None
    """
    retriever = _get_hybrid_retriever()
    if retriever:
        metrics = retriever.get_metrics()
        return metrics.to_dict() if metrics else None
    return None


def format_docs(docs: List[Document]) -> str:
    """将文档列表格式化为可读的上下文字符串

    将检索到的 Document 对象列表转换为格式化文本，
    便于 LLM 理解和引用。

    格式化格式：
    【参考资料 1】
    标题: xxx > yyy > zzz（如果有层级标题）
    来源: filename.md
    内容:
    [文档内容]

    【参考资料 2】
    ...

    Args:
        docs: Document 对象列表（来自向量检索）

    Returns:
        str: 格式化后的上下文字符串

    元数据来源：
    - _file_name: 原始文件名
    - h1, h2, h3: Markdown 标题（如果有）
    """
    formatted_parts = []

    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata

        # 获取来源文件名（用于显示）
        source = metadata.get("_file_name", "未知来源")

        # 提取标题层级信息
        # MarkdownHeaderTextSplitter 会将标题信息保存到 metadata
        # 格式如：{"h1": "第一章", "h2": "1.1 节"}
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])

        # 构建标题字符串（如 "第一章 > 1.1 节"）
        header_str = " > ".join(headers) if headers else ""

        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
