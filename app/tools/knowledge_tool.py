"""知识检索工具模块

从向量数据库（Milvus）中检索与用户问题相关的文档片段。
这是 RAG（检索增强生成）系统的核心工具。

工作原理：
1. 用户提出问题
2. 将问题嵌入为向量（embedding）
3. 在 Milvus 中检索最相似的 top-k 个文档块
4. 返回格式化的文档内容作为上下文
5. LLM 基于上下文生成回答

技术细节：
- 使用 LangChain VectorStore 的 as_retriever() 接口
- 检索参数：top_k = config.rag_top_k（默认 3）
- 返回格式：Tuple[str, List[Document]]（内容 + 原始文档）

@tool 装饰器参数：
- response_format="content_and_artifact": 返回值包含内容和原始对象两部分
  用于需要同时返回格式化文本和原始文档的场景
"""

from typing import List, Tuple  # 类型注解

from langchain_core.documents import Document  # LangChain 文档对象
from langchain_core.tools import tool         # LangChain 工具装饰器
from loguru import logger                    # 日志记录器

from app.config import config                          # 全局配置
from app.services.vector_store_manager import vector_store_manager  # 向量存储管理器


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，Agent 会自动调用此工具。

    调用场景示例：
    - 用户问："CPU 使用率过高怎么处理？"
    - Agent 分析：需要查询运维知识库中的相关文档
    - 调用 retrieve_knowledge("CPU 使用率过高")
    - 返回相关文档片段
    - Agent 基于文档内容生成回答

    工作流程：
    1. 获取 VectorStore 实例
    2. 创建检索器（配置 top_k）
    3. 调用检索器获取相关文档
    4. 格式化文档为字符串上下文
    5. 返回 (上下文文本, 原始文档列表)

    Args:
        query: 用户的查询文本
              通常是用户问题的原文或关键词
              示例："CPU 使用率过高怎么办"

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
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 获取 VectorStore 实例
        vector_store = vector_store_manager.get_vector_store()

        # 创建检索器
        # as_retriever() 将 VectorStore 转换为检索器接口
        # search_kwargs={"k": config.rag_top_k} 设置返回的文档数量
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )

        # 执行检索
        docs = retriever.invoke(query)

        # 无结果处理
        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        # 格式化文档为上下文字符串
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        # 返回错误信息，不抛出异常（让 Agent 能继续处理）
        return f"检索知识时发生错误: {str(e)}", []


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
