"""向量索引服务模块

将本地文件索引到向量数据库的服务。
这是 RAG 系统的入库端：
1. 读取文件内容（.txt、.md）
2. 分割文档为小块（chunk）
3. 将 chunk 添加到 Milvus

索引流程：
目录 → 文件 → 读取内容 → 分割文档 → 添加到向量存储

核心功能：
- index_directory(): 索引目录下所有支持的文件
- index_single_file(): 索引单个文件
"""

from datetime import datetime  # 时间记录，用于计算索引耗时
from pathlib import Path  # 路径操作
from typing import Any, Dict, Optional  # 类型注解

from loguru import logger  # 日志记录器

from app.services.document_splitter_service import document_splitter_service  # 文档分割服务
from app.services.vector_store_manager import vector_store_manager  # 向量存储管理器


class IndexingResult:
    """索引结果数据类

    封装索引操作的执行结果，包含：
    - 成功/失败状态
    - 文件统计（总数、成功数、失败数）
    - 耗时信息
    - 失败文件列表及错误原因
    """

    def __init__(self):
        """初始化索引结果（默认所有字段为空/零）"""
        self.success = False              # 是否全部成功
        self.directory_path = ""          # 被索引的目录路径
        self.total_files = 0              # 文件总数
        self.success_count = 0            # 成功数量
        self.fail_count = 0               # 失败数量
        self.start_time: Optional[datetime] = None   # 开始时间
        self.end_time: Optional[datetime] = None     # 结束时间
        self.error_message = ""           # 错误信息（如果有）
        self.failed_files: Dict[str, str] = {}  # 失败文件映射：路径 → 错误原因

    def increment_success_count(self):
        """增加成功计数"""
        self.success_count += 1

    def increment_fail_count(self):
        """增加失败计数"""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        """添加失败文件记录

        Args:
            file_path: 失败的文件路径
            error: 错误原因描述
        """
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """获取索引耗时（毫秒）

        Returns:
            int: 耗时毫秒数，如果未完成则返回 0
        """
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            可序列化为 JSON 的字典，包含所有索引统计信息
        """
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }


class VectorIndexService:
    """向量索引服务

    负责将文件系统中的文档索引到 Milvus 向量数据库。

    支持的文件格式：
    - .txt: 纯文本文件
    - .md: Markdown 文件（会保留标题结构进行智能分割）

    索引策略：
    1. 读取文件内容
    2. 删除该文件的旧数据（如果存在）
    3. 分割文档为小块
    4. 添加到向量存储（自动生成向量）
    """

    def __init__(self):
        """初始化向量索引服务"""
        # 默认上传目录（可通过 index_directory() 参数覆盖）
        self.upload_path = "./uploads"
        logger.info("向量索引服务初始化完成")

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """索引指定目录下的所有支持的文件

        Args:
            directory_path: 要索引的目录路径
                           如果为 None，使用默认的 upload_path

        Returns:
            IndexingResult: 索引结果，包含成功/失败统计和详细信息

        支持的文件：
        - *.txt: 纯文本文件
        - *.md: Markdown 文件
        """
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            # 确定目标目录
            target_path = directory_path if directory_path else self.upload_path
            dir_path = Path(target_path).resolve()

            # 验证目录有效性
            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在或不是有效目录: {target_path}")

            result.directory_path = str(dir_path)

            # 查找支持的文件（.txt 和 .md）
            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

            # 无文件时的处理
            if not files:
                logger.warning(f"目录中没有找到支持的文件: {target_path}")
                result.total_files = 0
                result.success = True  # 没有文件不算失败
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, 找到 {len(files)} 个文件")

            # 遍历索引每个文件
            for file_path in files:
                try:
                    self.index_single_file(str(file_path))
                    result.increment_success_count()
                    logger.info(f"✓ 文件索引成功: {file_path.name}")
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"✗ 文件索引失败: {file_path.name}, 错误: {e}")

            # 判断是否全部成功
            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result

    def index_single_file(self, file_path: str):
        """索引单个文件

        索引流程：
        1. 读取文件内容
        2. 删除该文件的旧数据（幂等性：支持重复索引）
        3. 分割文档为小块
        4. 添加到向量存储

        Args:
            file_path: 文件的绝对路径或相对路径

        Raises:
            ValueError: 文件不存在时抛出
            RuntimeError: 索引过程中发生错误时抛出
        """
        path = Path(file_path).resolve()

        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            # 步骤 1：读取文件内容
            content = path.read_text(encoding="utf-8")
            logger.info(f"读取文件: {path}, 内容长度: {len(content)} 字符")

            # 步骤 2：删除该文件的旧数据
            # 路径标准化为 POSIX 格式（使用正斜杠），确保一致性
            normalized_path = path.as_posix()
            vector_store_manager.delete_by_source(normalized_path)

            # 步骤 3：分割文档
            # document_splitter_service 会根据文件类型选择合适的分割策略
            documents = document_splitter_service.split_document(content, normalized_path)
            logger.info(f"文档分割完成: {file_path} -> {len(documents)} 个分片")

            # 步骤 4：添加到向量存储
            if documents:
                # vector_store_manager.add_documents 会自动：
                # 1. 为每个文档生成唯一 ID
                # 2. 调用 embedding_service 将文本转换为向量
                # 3. 批量插入 Milvus
                vector_store_manager.add_documents(documents)
                logger.info(f"文件索引完成: {file_path}, 共 {len(documents)} 个分片")
            else:
                logger.warning(f"文件内容为空或无法分割: {file_path}")

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e


# ========== 全局向量索引服务单例 ==========
vector_index_service = VectorIndexService()
