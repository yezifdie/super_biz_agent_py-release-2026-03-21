"""文档解析模块

提供多种文档格式的智能解析能力：
- MinerU 解析器：PDF/Word 版面分析、表格识别、公式识别
- Markdown 解析器：保留标题层级结构
- 纯文本解析器：简单文本提取

使用示例：
    from app.services.document_parser import ParserFactory

    parser = ParserFactory.create_parser("document.pdf")
    content = parser.parse("document.pdf")
"""

from app.services.document_parser.base_parser import BaseParser, ParseResult, ParserFactory
from app.services.document_parser.markdown_parser import MarkdownParser
from app.services.document_parser.text_parser import TextParser

# 尝试导入 MinerUParser（可能失败）
try:
    from app.services.document_parser.mineru_parser import MinerUParser
except ImportError:
    MinerUParser = None

# 注册解析器
_registered = False

def _ensure_parsers_registered():
    """确保解析器已注册（延迟注册避免循环导入）"""
    global _registered
    if _registered:
        return

    from app.services.document_parser.markdown_parser import MarkdownParser
    from app.services.document_parser.text_parser import TextParser

    ParserFactory.register(['.md', '.markdown'], MarkdownParser)
    ParserFactory.register(['.txt', '.text'], TextParser)

    if MinerUParser is not None:
        try:
            parser = MinerUParser()
            ParserFactory.register(['.pdf', '.docx', '.doc'], MinerUParser)
        except Exception:
            pass  # MinerU 不可用

    _registered = True


def create_parser(file_path: str) -> BaseParser:
    """创建合适的解析器

    Args:
        file_path: 文件路径

    Returns:
        BaseParser: 解析器实例
    """
    _ensure_parsers_registered()
    return ParserFactory.create_parser(file_path)


def parse(file_path: str) -> ParseResult:
    """直接解析文档

    Args:
        file_path: 文件路径

    Returns:
        ParseResult: 解析结果
    """
    parser = create_parser(file_path)
    return parser.parse(file_path)


def get_supported_extensions() -> list:
    """获取支持的扩展名"""
    _ensure_parsers_registered()
    return ParserFactory.get_supported_extensions()


__all__ = [
    "BaseParser",
    "ParseResult",
    "ParserFactory",
    "MinerUParser",
    "MarkdownParser",
    "TextParser",
    "create_parser",
    "parse",
    "get_supported_extensions",
]
