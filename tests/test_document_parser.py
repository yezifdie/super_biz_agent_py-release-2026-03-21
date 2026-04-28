"""文档解析器单元测试"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import tempfile
from pathlib import Path


class TestParserFactory(unittest.TestCase):
    """解析器工厂测试"""

    def test_create_parser_for_markdown(self):
        """测试为 Markdown 文件创建解析器"""
        from app.services.document_parser import ParserFactory, MarkdownParser
        
        parser = ParserFactory.create_parser("test.md")
        self.assertIsInstance(parser, MarkdownParser)

    def test_create_parser_for_text(self):
        """测试为文本文件创建解析器"""
        from app.services.document_parser import ParserFactory, TextParser
        
        parser = ParserFactory.create_parser("test.txt")
        self.assertIsInstance(parser, TextParser)

    def test_supported_extensions(self):
        """测试支持的扩展名"""
        from app.services.document_parser import ParserFactory
        
        extensions = ParserFactory.get_supported_extensions()
        self.assertIn('.md', extensions)
        self.assertIn('.txt', extensions)


class TestMarkdownParser(unittest.TestCase):
    """Markdown 解析器测试"""

    def setUp(self):
        """测试前准备"""
        from app.services.document_parser import MarkdownParser
        self.parser = MarkdownParser()
        
        # 创建临时文件
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.md")

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_parse_simple_markdown(self):
        """测试解析简单 Markdown"""
        content = """# 标题

这是正文内容。

## 子标题

这是子标题下的内容。
"""
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        
        self.assertIsNotNone(result)
        self.assertIn("标题", result.content)
        self.assertIn("正文", result.content)

    def test_parse_with_code_block(self):
        """测试解析代码块"""
        content = """# Python 示例

```python
def hello():
    print("Hello")
```
"""
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)

        self.assertIn("def hello", result.content)
        self.assertGreater(result.metadata.get("code_block_count", 0), 0)

    def test_parse_with_table(self):
        """测试解析表格"""
        content = """| 列1 | 列2 |
| --- | --- |
| A1 | A2 |
| B1 | B2 |
"""
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        
        self.assertIn("列1", result.content)
        self.assertIn("A1", result.content)

    def test_metadata_extraction(self):
        """测试元数据提取"""
        content = """# 文档标题

这是正文。
"""
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        
        self.assertEqual(result.metadata.get("file_type"), "markdown")
        self.assertEqual(result.metadata.get("parser"), "MarkdownParser")
        self.assertIn("heading_count", result.metadata)

    def test_file_not_found(self):
        """测试文件不存在"""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("non_existent_file.md")

    def test_supports_method(self):
        """测试 supports 方法"""
        self.assertTrue(self.parser.supports("test.md"))
        self.assertTrue(self.parser.supports("test.markdown"))
        self.assertFalse(self.parser.supports("test.pdf"))


class TestTextParser(unittest.TestCase):
    """文本解析器测试"""

    def setUp(self):
        """测试前准备"""
        from app.services.document_parser import TextParser
        self.parser = TextParser()
        
        # 创建临时文件
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test.txt")

    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_parse_simple_text(self):
        """测试解析纯文本"""
        content = "这是第一行\n这是第二行\n这是第三行"
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        
        self.assertEqual(result.content, content)
        self.assertEqual(result.metadata.get("file_type"), "text")

    def test_metadata_extraction(self):
        """测试元数据提取"""
        content = "测试内容"
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        
        self.assertIn("line_count", result.metadata)
        self.assertIn("char_count", result.metadata)

    def test_utf8_encoding(self):
        """测试 UTF-8 编码"""
        content = "中文内容 English content 日本語"
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        result = self.parser.parse(self.temp_file)
        self.assertIn("中文", result.content)
        self.assertIn("English", result.content)


class TestMinerUParser(unittest.TestCase):
    """MinerU 解析器测试"""

    def test_parser_creation(self):
        """测试解析器创建"""
        from app.services.document_parser import MinerUParser
        
        parser = MinerUParser()
        self.assertIsNotNone(parser)

    def test_supports_pdf(self):
        """测试支持 PDF"""
        from app.services.document_parser import MinerUParser
        
        parser = MinerUParser()
        self.assertTrue(parser.supports("test.pdf"))
        self.assertTrue(parser.supports("test.PDF"))

    def test_supports_docx(self):
        """测试支持 Word"""
        from app.services.document_parser import MinerUParser
        
        parser = MinerUParser()
        self.assertTrue(parser.supports("test.docx"))
        self.assertTrue(parser.supports("test.doc"))

    def test_supports_unsupported(self):
        """测试不支持的格式"""
        from app.services.document_parser import MinerUParser
        
        parser = MinerUParser()
        self.assertFalse(parser.supports("test.xlsx"))
        self.assertFalse(parser.supports("test.pptx"))

    def test_get_status(self):
        """测试获取状态"""
        from app.services.document_parser import MinerUParser
        
        parser = MinerUParser()
        status = parser.get_status()
        
        self.assertIn("available", status)
        self.assertIn("gpu_available", status)
        self.assertIn("magic_pdf_available", status)


class TestParseResult(unittest.TestCase):
    """解析结果测试"""

    def test_parse_result_to_dict(self):
        """测试 ParseResult 转换为字典"""
        from app.services.document_parser.base_parser import ParseResult
        
        result = ParseResult(
            content="测试内容",
            metadata={"key": "value"},
            warnings=["warning1"]
        )
        
        d = result.to_dict()
        
        self.assertEqual(d["content"], "测试内容")
        self.assertEqual(d["metadata"]["key"], "value")
        self.assertIn("warning1", d["warnings"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
