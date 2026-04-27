"""文档相关数据模型模块

定义文档处理流程中的数据结构。
主要涉及文档上传后的切分（Splitting）操作，将大文档拆成小片段以便检索。
"""

from typing import Optional  # Optional: 标记可选字段（可为 None）

from pydantic import BaseModel, Field  # BaseModel: Pydantic 模型基类，Field: 字段元数据


class DocumentChunk(BaseModel):
    """文档分片模型

    表示文档被切分后的一个片段（Chunk）。
    在 RAG 系统中，文档被切成多个 chunk 后分别生成向量存入向量数据库。
    检索时，根据向量相似度找到最相关的 top-k 个 chunk，拼接成上下文给 LLM。

    为什么需要分片？
    1. 控制单次检索的精度：太长的文本会让语义变得模糊
    2. 节省 token：LLM 的上下文窗口有限，需要精打8a39细算
    3. 提高检索粒度：用户问题通常针对某个具体点，精准匹配比模糊匹配好

    分片策略考量：
    - chunk_max_size: 每个块的最大字符数（配置中默认 800）
    - chunk_overlap: 相邻块之间的重叠字符数（默认 100）
      重叠的目的是避免在切分点丢失语义连贯性

    示例：
        假设原文档内容为："第一章 概述。本章介绍系统架构。第二章 原理..."
        分片后可能是：
        Chunk 0: {content: "第一章 概述。本章介绍系统架构。", start: 0, end: 20, index: 0, title: "第一章"}
        Chunk 1: {content: "本章介绍系统架构。第二章 原理...", start: 15, end: 50, index: 1, title: "第一章"}
    """

    # 分片的具体文本内容
    # 这是将被向量化并存储的核心数据
    content: str = Field(..., description="分片内容")

    # 分片在原文档中的字符起始位置
    # 用于追溯原始文档、维护文档顺序、计算重叠区域
    start_index: int = Field(..., description="分片在原文档中的起始位置")

    # 分片在原文档中的字符结束位置（不包含，即 [start_index, end_index)）
    end_index: int = Field(..., description="分片在原文档中的结束位置")

    # 分片在文档中的顺序索引（从 0 开始）
    # 用于重建文档顺序、显示分片位置信息
    chunk_index: int = Field(..., description="分片索引（从0开始）")

    # 分片所属的章节/段落标题（可选）
    # 用于提供更多语义信息给向量检索
    # 如果原文档有标题结构，可以提取出来增加检索准确度
    title: Optional[str] = Field(None, description="分片所属章节标题")

    class Config:
        """Pydantic 配置"""

        # OpenAPI 文档示例
        json_schema_extra = {
            "example": {
                "content": "这是一段文档内容...",
                "start_index": 0,
                "end_index": 100,
                "chunk_index": 0,
                "title": "第一章",
            }
        }
