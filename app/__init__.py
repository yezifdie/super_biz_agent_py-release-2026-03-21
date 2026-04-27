"""SuperBizAgent Python 版本

基于 LangChain 的智能业务代理系统
"""

# 语义化版本号，用于版本管理和 API 兼容性判断
__version__ = "1.0.0"

# 从 app.utils 模块导入 logger 并注册到当前包的命名空间
# 这样其他模块可以直接通过 from app import logger 导入使用
# noqa: F401 是 flake8 忽略"导入但未使用"警告的指令，因为这里的目的是导出
from app.utils import logger  # noqa: F401
