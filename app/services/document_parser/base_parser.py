"""文档解析器基类和工厂模块

定义解析器接口规范和工厂方法，统一管理不同类型文档的解析逻辑。

设计模式：
- 策略模式：根据文档类型选择不同的解析策略
- 工厂模式：统一创建解析器实例
- 适配器模式：将不同解析器的输出统一为 Markdown 格式
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from loguru import logger


@dataclass
class ParseResult:
    """文档解析结果

    统一格式的解析输出，包含：
    - content: Markdown 格式的文本内容
    - metadata: 文档元数据（标题、页数、表格数等）
    - warnings: 解析过程中的警告信息
    """
    content: str                          # Markdown 格式内容
    metadata: Dict[str, Any]              # 解析元数据
    warnings: List[str]                    # 警告信息列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


class BaseParser(ABC):
    """文档解析器基类
    
    所有解析器必须实现的接口规范。
    
    接口要求：
    1. parse() 方法：执行文档解析
    2. supports() 方法：判断是否支持该文件类型
    3. extract_metadata() 方法：提取文档元数据
    """
    
    @abstractmethod
    def parse(self, file_path: str) -> ParseResult:
        """解析文档并返回 Markdown 内容
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            ParseResult: 解析结果，包含 Markdown 内容和元数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
            RuntimeError: 解析过程出错
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """判断是否支持该文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持
        """
        pass
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取文档元数据
        
        默认实现返回基本信息，子类可重写实现更详细的元数据提取。
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 元数据字典
        """
        path = Path(file_path)
        return {
            "file_name": path.name,
            "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            "file_type": path.suffix.lower(),
            "parser": self.__class__.__name__,
        }
    
    def validate_file(self, file_path: str) -> None:
        """验证文件有效性
        
        Args:
            file_path: 文件路径
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not self.supports(file_path):
            raise ValueError(f"解析器 {self.__class__.__name__} 不支持文件类型: {Path(file_path).suffix}")
    
    def _sanitize_content(self, content: str) -> str:
        """清理文档内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 清理后的内容
        """
        # 移除多余的空白字符
        content = '\n'.join(line.rstrip() for line in content.splitlines())
        # 移除连续的空行
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        return content.strip()


class ParserFactory:
    """解析器工厂类
    
    根据文件类型自动选择合适的解析器。
    
    使用示例：
        parser = ParserFactory.create_parser("document.pdf")
        result = parser.parse("document.pdf")
        
        # 或直接解析
        result = ParserFactory.parse("document.pdf")
    """
    
    # 解析器注册表：文件扩展名 -> 解析器类
    _parsers: Dict[str, type[BaseParser]] = {}
    
    @classmethod
    def register(cls, extensions: List[str], parser_class: type[BaseParser]) -> None:
        """注册解析器
        
        Args:
            extensions: 支持的文件扩展名列表，如 ['.pdf', '.docx']
            parser_class: 解析器类
        """
        for ext in extensions:
            ext = ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
            cls._parsers[ext] = parser_class
            logger.debug(f"注册解析器: {ext} -> {parser_class.__name__}")
    
    @classmethod
    def create_parser(cls, file_path: str) -> BaseParser:
        """创建合适的解析器

        Args:
            file_path: 文件路径

        Returns:
            BaseParser: 解析器实例

        Raises:
            ValueError: 没有找到支持该文件类型的解析器
        """
        # 确保解析器已注册
        if not cls._parsers:
            cls._do_register()

        ext = Path(file_path).suffix.lower()

        if ext not in cls._parsers:
            # 尝试降级到纯文本解析器
            logger.warning(f"未找到 {ext} 的专用解析器，降级到纯文本解析器")
            from app.services.document_parser.text_parser import TextParser
            return TextParser()

        parser_class = cls._parsers[ext]
        return parser_class()

    @classmethod
    def _do_register(cls) -> None:
        """执行解析器注册"""
        from app.services.document_parser.markdown_parser import MarkdownParser
        from app.services.document_parser.text_parser import TextParser

        ParserFactory.register(['.md', '.markdown'], MarkdownParser)
        ParserFactory.register(['.txt', '.text'], TextParser)

        try:
            from app.services.document_parser.mineru_parser import MinerUParser
            try:
                parser = MinerUParser()
                ParserFactory.register(['.pdf', '.docx', '.doc'], MinerUParser)
            except Exception:
                pass
        except ImportError:
            pass
    
    @classmethod
    def parse(cls, file_path: str) -> ParseResult:
        """直接解析文档（便捷方法）
        
        Args:
            file_path: 文件路径
            
        Returns:
            ParseResult: 解析结果
        """
        parser = cls.create_parser(file_path)
        return parser.parse(file_path)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取所有支持的文件扩展名
        
        Returns:
            List[str]: 扩展名列表
        """
        return list(cls._parsers.keys())


# 注册默认解析器
# 每个解析器模块应该导入后自行注册
# 基础模块不注册任何解析器，由 __init__.py 统一注册
