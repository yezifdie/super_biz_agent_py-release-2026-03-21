"""Markdown 文档解析模块

解析 Markdown 格式的文档，保留标题层级结构和代码块。

特性：
- 标题识别：自动识别 # ~ ###### 标题层级
- 代码块保留：保留代码块的原始格式
- 链接处理：保留链接但清理外部 URL
- 列表处理：保留有序和无序列表

使用示例：
    from app.services.document_parser import MarkdownParser
    
    parser = MarkdownParser()
    result = parser.parse("document.md")
    print(result.content)
"""

import os
from pathlib import Path
from typing import Dict, List, Any

from loguru import logger

from app.services.document_parser.base_parser import BaseParser, ParseResult


class MarkdownParser(BaseParser):
    """Markdown 文档解析器
    
    专门处理 Markdown 格式的文档文件。
    自动保留文档结构和层级关系。
    """
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = ['.md', '.markdown']
    
    def __init__(self):
        """初始化 Markdown 解析器"""
        super().__init__()
        self._heading_pattern = None
        self._link_pattern = None
    
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
        """解析 Markdown 文档
        
        Args:
            file_path: Markdown 文件路径
            
        Returns:
            ParseResult: 解析结果
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        import re
        
        # 验证文件
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not self.supports(file_path):
            raise ValueError(f"MarkdownParser 不支持文件类型: {Path(file_path).suffix}")
        
        warnings = []
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取元数据
            metadata = self._extract_metadata(content, file_path)
            
            # 预处理内容
            content = self._preprocess_content(content)
            
            # 验证 Markdown 结构
            warnings.extend(self._validate_structure(content))
            
            # 清理内容
            content = self._sanitize_content(content)
            
            logger.info(
                f"Markdown 解析完成: {metadata.get('heading_count', 0)} 个标题, "
                f"{metadata.get('code_block_count', 0)} 个代码块, "
                f"{len(content)} 字符"
            )
            
            return ParseResult(
                content=content,
                metadata=metadata,
                warnings=warnings
            )
            
        except UnicodeDecodeError:
            logger.error(f"文件编码错误: {file_path}")
            warnings.append("文件编码错误，尝试 GBK 编码")
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return ParseResult(
                    content=self._sanitize_content(content),
                    metadata=self._extract_metadata(content, file_path),
                    warnings=warnings
                )
            except Exception:
                raise RuntimeError(f"无法读取文件: {file_path}")
        except Exception as e:
            logger.error(f"Markdown 解析失败: {e}")
            raise RuntimeError(f"Markdown 解析失败: {e}") from e
    
    def _extract_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """提取 Markdown 文档元数据
        
        Args:
            content: 文档内容
            file_path: 文件路径
            
        Returns:
            Dict: 元数据字典
        """
        import re
        
        metadata = {
            "file_name": Path(file_path).name,
            "file_type": "markdown",
            "parser": "MarkdownParser",
        }
        
        # 统计标题数量
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headings = heading_pattern.findall(content)
        metadata['heading_count'] = len(headings)
        
        # 统计各层级标题
        heading_levels = {}
        for marker, _ in headings:
            level = len(marker)
            heading_levels[level] = heading_levels.get(level, 0) + 1
        metadata['heading_levels'] = heading_levels
        
        # 统计代码块
        code_block_pattern = re.compile(r'```[\s\S]*?```|`[^`]+`')
        code_blocks = code_block_pattern.findall(content)
        metadata['code_block_count'] = len(code_blocks)
        
        # 统计表格
        table_pattern = re.compile(r'\|.+\|')
        tables = table_pattern.findall(content)
        metadata['table_count'] = len(set(tables)) // 3 if tables else 0  # 每表3行
        
        # 统计链接
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')
        links = link_pattern.findall(content)
        metadata['link_count'] = len(links)
        
        # 统计列表
        list_pattern = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
        ul_items = list_pattern.findall(content)
        ol_pattern = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
        ol_items = ol_pattern.findall(content)
        metadata['list_count'] = len(ul_items) + len(ol_items)
        
        # 提取文档标题（第一个 H1）
        first_h1 = heading_pattern.search(content)
        if first_h1:
            metadata['title'] = first_h1.group(2).strip()
        
        metadata['content_length'] = len(content)
        
        return metadata
    
    def _preprocess_content(self, content: str) -> str:
        """预处理 Markdown 内容
        
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
        
        # 清理 HTML 标签（可选）
        # html_pattern = re.compile(r'<[^>]+>')
        # content = html_pattern.sub('', content)
        
        # 标准化链接（移除外部 URL 中的跟踪参数）
        link_pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\?)]+)\?[^)]*\)')
        content = link_pattern.sub(r'[\1](\2)', content)
        
        return content
    
    def _validate_structure(self, content: str) -> List[str]:
        """验证 Markdown 结构
        
        Args:
            content: 文档内容
            
        Returns:
            List[str]: 警告信息列表
        """
        import re
        warnings = []
        
        # 检查标题层级是否跳跃（如 H1 后直接 H3）
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        headings = heading_pattern.findall(content)
        
        if headings:
            prev_level = 0
            for marker, text in headings:
                level = len(marker)
                if level - prev_level > 1 and prev_level > 0:
                    warnings.append(
                        f"标题层级跳跃: H{prev_level} 后直接出现 H{level}"
                    )
                prev_level = level
        
        # 检查表格格式
        lines = content.split('\n')
        in_table = False
        table_line_count = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('|'):
                if not in_table:
                    in_table = True
                    table_line_count = 1
                else:
                    table_line_count += 1
            else:
                if in_table and table_line_count > 0:
                    # 检查是否是有效的表格（至少3行）
                    if table_line_count < 3:
                        warnings.append(
                            f"表格格式可能不正确（第 {i - table_line_count + 1} 行附近）"
                        )
                    in_table = False
        
        return warnings
    
    def _sanitize_content(self, content: str) -> str:
        """清理 Markdown 内容
        
        Args:
            content: 原始内容
            
        Returns:
            str: 清理后的内容
        """
        if not content:
            return ""
        
        # 移除多余的空白字符
        lines = []
        for line in content.split('\n'):
            lines.append(line.rstrip())
        
        content = '\n'.join(lines)
        
        # 移除连续的空行（最多保留两个）
        while '\n\n\n' in content:
            content = content.replace('\n\n\n', '\n\n')
        
        return content.strip()
