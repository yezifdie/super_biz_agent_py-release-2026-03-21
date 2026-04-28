"""纯文本文档解析模块

解析纯文本格式的文档，进行简单的格式化和清理。

使用示例：
    from app.services.document_parser import TextParser
    
    parser = TextParser()
    result = parser.parse("document.txt")
    print(result.content)
"""

import os
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger

from app.services.document_parser.base_parser import BaseParser, ParseResult


class TextParser(BaseParser):
    """纯文本文档解析器
    
    处理 .txt 等纯文本格式文件，进行简单的格式化和清理。
    """
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = ['.txt', '.text', '.log', '.csv']
    
    def supports(self, file_path: str) -> bool:
        """判断是否支持该文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def parse(self, file_path: str) -> ParseResult:
        """解析纯文本文档
        
        Args:
            file_path: 文本文件路径
            
        Returns:
            ParseResult: 解析结果
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        # 验证文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not self.supports(file_path):
            raise ValueError(f"TextParser 不支持文件类型: {Path(file_path).suffix}")
        
        warnings = []
        
        try:
            # 尝试 UTF-8 编码
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试 GBK 编码
                warnings.append("UTF-8 解码失败，尝试 GBK 编码")
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # 尝试 GB18030 编码
                warnings.append("GBK 解码失败，尝试 GB18030 编码")
                with open(file_path, 'r', encoding='gb18030') as f:
                    content = f.read()
            
            # 提取元数据
            metadata = self._extract_metadata(content, file_path)
            
            # 预处理
            content = self._preprocess_content(content)
            
            # 清理
            content = self._sanitize_content(content)
            
            logger.info(
                f"文本解析完成: {len(content)} 字符, "
                f"行数: {metadata.get('line_count', 0)}"
            )
            
            return ParseResult(
                content=content,
                metadata=metadata,
                warnings=warnings
            )
            
        except FileNotFoundError:
            raise
        except UnicodeDecodeError:
            logger.error(f"文件编码不支持: {file_path}")
            raise ValueError(f"无法解码文件，请确保文件编码为 UTF-8、GBK 或 GB18030")
        except Exception as e:
            logger.error(f"文本解析失败: {e}")
            raise RuntimeError(f"文本解析失败: {e}") from e
    
    def _extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """提取文本元数据
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            Dict: 元数据字典
        """
        # 基本统计
        line_count = content.count('\n') + 1
        word_count = len(content)  # 中文字符直接计数
        char_count = len(content)
        
        # 估算英文单词数
        import re
        english_words = len(re.findall(r'[a-zA-Z]+', content))
        word_count = char_count + english_words // 2  # 粗略估算
        
        # 行数统计
        lines = content.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        # 检测是否有表格
        table_like_lines = [
            i for i, l in enumerate(lines) 
            if l.strip().count('\t') > 2 or l.strip().count('|') > 2
        ]
        
        metadata = {
            "file_name": Path(file_path).name,
            "file_type": "text",
            "parser": "TextParser",
            "line_count": line_count,
            "non_empty_line_count": len(non_empty_lines),
            "char_count": char_count,
            "word_count": word_count,
            "has_table_like_content": len(table_like_lines) > 0,
            "content_length": len(content),
        }
        
        return metadata
    
    def _preprocess_content(self, content: str) -> str:
        """预处理文本内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 预处理后的内容
        """
        import re
        
        # 移除 BOM
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # 标准化换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 标准化制表符（转换为4个空格）
        content = content.replace('\t', '    ')
        
        # 移除控制字符（保留换行和制表符）
        content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
        
        return content
    
    def _sanitize_content(self, content: str) -> str:
        """清理文本内容
        
        Args:
            content: 预处理后的内容
            
        Returns:
            str: 清理后的内容
        """
        if not content:
            return ""
        
        # 移除首尾空白
        content = content.strip()
        
        # 移除行尾空白
        lines = []
        for line in content.split('\n'):
            lines.append(line.rstrip())
        
        content = '\n'.join(lines)
        
        # 移除连续的空行（最多保留两个）
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        
        return content
