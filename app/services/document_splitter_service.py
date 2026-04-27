"""文档分割服务模块

将长文档分割成小块（Chunk）的服务。
这是 RAG 系统中的关键步骤，分割质量直接影响检索效果。

分割策略设计原则：
1. 语义完整性：尽量保持每个 chunk 的语义完整（不截断句子、段落）
2. 大小可控：每个 chunk 不能太大（影响向量精度和上下文利用率）
3. 重叠保序：相邻 chunk 之间有重叠，保持上下文连贯性

LangChain 提供的分割器：
- MarkdownHeaderTextSplitter: 按 Markdown 标题分割（保留文档结构）
- RecursiveCharacterTextSplitter: 按字符递归分割（保证 chunk 大小）
"""

from pathlib import Path  # 路径操作（提取文件名、后缀等）
from typing import List  # 类型注解

from langchain_core.documents import Document  # LangChain 文档对象
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,  # Markdown 标题分割器
    RecursiveCharacterTextSplitter,  # 递归字符分割器
)
from loguru import logger  # 日志记录器

from app.config import config  # 全局配置


class DocumentSplitterService:
    """文档分割服务

    提供多种分割策略：
    - split_markdown(): Markdown 文档专用（两阶段分割 + 小片段合并）
    - split_text(): 普通文本分割
    - split_document(): 智能选择分割策略（根据文件类型）

    分割流程（Markdown）：
    阶段1: MarkdownHeaderTextSplitter → 按 # 和 ## 标题分割
    阶段2: RecursiveCharacterTextSplitter → 按字符数进一步分割
    阶段3: _merge_small_chunks() → 合并过小的片段（< 300 字符）
    """

    def __init__(self):
        """初始化文档分割服务

        从全局配置读取分割参数：
        - chunk_max_size: 每个 chunk 的最大字符数（默认 800）
        - chunk_overlap: 相邻 chunk 的重叠字符数（默认 100）
        """
        self.chunk_size = config.chunk_max_size       # 最大 chunk 大小
        self.chunk_overlap = config.chunk_overlap     # chunk 重叠大小

        # Markdown 标题分割器：按 # 和 ## 分割
        # 只分割到二级标题，避免过度碎片化
        # strip_headers=False: 在内容中保留标题，保持语义完整
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),   # 一级标题分割
                ("##", "h2"),  # 二级标题分割
                # 注意：不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器：用于二次分割
        # chunk_size * 2 是为了让每个 chunk 更大一些，减少分片数
        # 这样可以保留更多上下文
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,   # 加倍 chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap, # 重叠保持语义连贯
            length_function=len,              # 使用字符数作为长度函数
            is_separator_regex=False,         # 分隔符不使用正则表达式
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """分割 Markdown 文档（两阶段分割 + 小片段合并）

        适用场景：.md 文件，保持 Markdown 标题结构

        分割流程：
        1. MarkdownHeaderTextSplitter → 按标题分割（保留文档结构）
        2. RecursiveCharacterTextSplitter → 按字符数分割（控制大小）
        3. _merge_small_chunks() → 合并过小的片段（保持 chunk 有意义）

        Args:
            content: Markdown 文档的文本内容
            file_path: 文件路径（用于元数据）

        Returns:
            List[Document]: 分割后的 Document 列表

        示例：
            # 输入
            "# 第一章\n这是第一章的内容。\n## 1.1 节\n这是 1.1 节的内容。"
            # 输出（假设 chunk_size=800）
            [Document("第一章\n这是第一章的内容。", metadata={...}),
             Document("## 1.1 节\n这是 1.1 节的内容。", metadata={...})]
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段：按 Markdown 标题分割
            # 分割点：# 和 ## 标题
            # 优点：保持标题与内容的关联
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段：按字符数进一步分割
            # 优点：控制每个 chunk 的大小
            # 注意：只会对超过 chunk_size * 2 的 chunk 进行分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段：合并太小的分片
            # 原因：过小的 chunk（如只有标题的几行字）语义信息不足
            # 策略：合并 < 300 字符的小片段到前一个 chunk
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path           # 原始文件路径
                doc.metadata["_extension"] = ".md"            # 文件扩展名
                doc.metadata["_file_name"] = Path(file_path).name  # 文件名

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """分割普通文本文档

        适用场景：.txt 文件，无特殊结构

        直接使用 RecursiveCharacterTextSplitter 分割，
        按字符数控制大小。

        Args:
            content: 文本内容
            file_path: 文件路径（用于元数据）

        Returns:
            List[Document]: 分割后的 Document 列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 使用 text_splitter 创建文档
            # create_documents() 接受文本列表和元数据列表
            docs = self.text_splitter.create_documents(
                texts=[content],  # 文本内容（列表）
                metadatas=[  # 元数据列表（与文本一一对应）
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,  # 文件后缀
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """智能分割文档（根据文件类型选择分割器）

        根据文件扩展名自动选择合适的分割策略：
        - .md → split_markdown()（保留标题结构）
        - 其他 → split_text()（通用文本分割）

        Args:
            content: 文档内容
            file_path: 文件路径（用于判断文件类型和元数据）

        Returns:
            List[Document]: 分割后的 Document 列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """合并太小的分片

        策略：
        - 遍历文档列表
        - 如果当前文档 < min_size 字符 且 合并后不会太大 → 合并到前一个文档
        - 否则 → 保存当前文档，开始新文档

        Args:
            documents: 分割后的文档列表
            min_size: 最小 chunk 大小（字符数）

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档，直接作为当前文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # 当前文档太小，且合并后不会超过上限
                # 合并：追加到当前文档内容
                current_doc.page_content += "\n\n" + doc.page_content
                # 元数据保留主文档的（第一个文档的）
            else:
                # 当前文档足够大，或合并后会太大
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


# ========== 全局文档分割服务单例 ==========
document_splitter_service = DocumentSplitterService()
