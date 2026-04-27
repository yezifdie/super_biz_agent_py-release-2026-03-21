"""文件上传接口模块

提供文档上传和索引管理的 API：
1. upload_file(): 上传文件并自动创建向量索引
2. index_directory(): 索引指定目录下的所有文件

支持的文档格式：
- .txt: 纯文本文件
- .md: Markdown 文件（会保留标题结构进行智能分割）

文件大小限制：10MB
"""

from pathlib import Path  # 路径操作

from fastapi import APIRouter, File, HTTPException, UploadFile  # FastAPI 文件上传
from fastapi.responses import JSONResponse  # JSON 响应

from app.services.vector_index_service import vector_index_service  # 向量索引服务
from loguru import logger  # 日志记录器

# 创建 APIRouter
router = APIRouter()

# 上传文件存储目录
UPLOAD_DIR = Path("./uploads")

# 支持的文件扩展名
ALLOWED_EXTENSIONS = ["txt", "md"]

# 单个文件最大大小（10MB）
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并自动创建向量索引

    上传文档后，系统会自动：
    1. 保存文件到服务器
    2. 读取文件内容
    3. 分割文档为小块
    4. 生成向量并存储到 Milvus

    请求格式：multipart/form-data
    - file: 二进制文件内容

    响应示例：
        {
            "code": 200,
            "message": "success",
            "data": {
                "filename": "cpu_high_usage.md",
                "file_path": "./uploads/cpu_high_usage.md",
                "size": 12345
            }
        }

    Args:
        file: UploadFile，FastAPI 自动处理文件上传
              File(...) 表示这是必需的请求参数

    Returns:
        JSONResponse: 上传结果

    Raises:
        HTTPException: 400（文件格式不支持、大小超限）、500（服务器错误）
    """
    try:
        # 步骤 1：验证文件名
        if not file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")

        # 步骤 2：规范化文件名（处理空格和特殊字符）
        safe_filename = _sanitize_filename(file.filename)

        # 步骤 3：验证文件扩展名
        file_extension = _get_file_extension(safe_filename)
        if file_extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # 步骤 4：创建上传目录
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 步骤 5：保存文件
        file_path = UPLOAD_DIR / safe_filename

        # 如果文件已存在，先删除旧文件（实现覆盖更新）
        if file_path.exists():
            logger.info(f"文件已存在，将覆盖: {file_path}")
            file_path.unlink()

        # 读取文件内容
        content = await file.read()

        # 步骤 6：验证文件大小
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制（最大 {MAX_FILE_SIZE} 字节）"
            )

        # 保存文件到磁盘
        file_path.write_bytes(content)

        logger.info(f"文件上传成功: {file_path}")

        # 步骤 7：自动创建向量索引
        try:
            logger.info(f"开始为上传文件创建向量索引: {file_path}")
            vector_index_service.index_single_file(str(file_path))
            logger.info(f"向量索引创建成功: {file_path}")
        except Exception as e:
            # 即使索引失败，文件上传仍然成功
            logger.error(f"向量索引创建失败: {file_path}, 错误: {e}")

        # 步骤 8：返回响应
        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success",
                "data": {
                    "filename": safe_filename,
                    "file_path": str(file_path),
                    "size": len(content),
                },
            },
        )

    except HTTPException:
        # HTTP 异常直接重新抛出
        raise
    except Exception as e:
        logger.error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {e}")


@router.post("/index_directory")
async def index_directory(directory_path: str = None):
    """索引指定目录下的所有文件

    批量索引目录中的 .txt 和 .md 文件。
    每个文件都会被读取、分割、生成向量并存储。

    请求示例：
        POST /api/index_directory
        Content-Type: application/x-www-form-urlencoded

        directory_path=./aiops-docs

    响应示例：
        {
            "code": 200,
            "message": "success",
            "data": {
                "success": true,
                "directory_path": "./aiops-docs",
                "total_files": 5,
                "success_count": 5,
                "fail_count": 0,
                "duration_ms": 1234,
                "error_message": "",
                "failed_files": {}
            }
        }

    Args:
        directory_path: 要索引的目录路径（可选，默认使用 uploads 目录）

    Returns:
        JSONResponse: 索引结果

    Raises:
        HTTPException: 500 错误时抛出
    """
    try:
        logger.info(f"开始索引目录: {directory_path or 'uploads'}")

        # 执行索引
        result = vector_index_service.index_directory(directory_path)

        return JSONResponse(
            status_code=200,
            content={
                "code": 200,
                "message": "success" if result.success else "partial_success",
                "data": result.to_dict(),
            },
        )

    except Exception as e:
        logger.error(f"索引目录失败: {e}")
        raise HTTPException(status_code=500, detail=f"索引目录失败: {e}")


def _get_file_extension(filename: str) -> str:
    """获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        str: 扩展名（小写，不含点），如 "txt", "md"
    """
    parts = filename.rsplit(".", 1)  # 从右边分割，只分一次
    if len(parts) == 2:
        return parts[1].lower()  # 返回小写的扩展名
    return ""


def _sanitize_filename(filename: str) -> str:
    """规范化文件名

    替换文件名中的空格和特殊字符，避免文件系统问题。
    适用于：Windows 和 Linux/macOS

    Args:
        filename: 原始文件名

    Returns:
        str: 规范化后的文件名

    示例：
        "my document.txt" → "my_document.txt"
        "file:name.md" → "file_name.md"
    """
    # 替换空格为下划线
    sanitized = filename.replace(" ", "_")
    # 替换 Windows/macOS/Linux 文件名中的非法字符
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized
