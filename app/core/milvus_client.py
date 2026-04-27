"""Milvus 向量数据库客户端管理模块

Milvus 是开源的向量数据库，专门用于存储和检索高维向量。
本模块封装了 Milvus 的连接管理、Collection 创建、索引构建等操作。

核心概念：
- Collection（集合）：类似于关系型数据库中的表，用于存储向量数据
- Vector（向量）：高维数组，通常由文本通过 embedding 模型生成
- Field（字段）：Collection 中的列，定义数据类型
- Index（索引）：加速向量检索的数据结构
- Shard（分片）：数据分区，用于分布式存储和并行查询

设计要点：
1. 单例模式：应用共享一个全局的 Milvus 连接
2. 幂等性：connect() 方法可安全重复调用
3. 自动初始化：启动时自动创建 Collection 和索引
4. 上下文管理器：支持 with 语句自动关闭连接
5. ORM 兼容性修复：解决 langchain-milvus 与 pymilvus ORM 的别名冲突问题

使用示例：
    # 方式 1：作为上下文管理器（推荐，自动管理生命周期）
    with milvus_manager:
        collection = milvus_manager.get_collection()
        # ... 使用 collection ...

    # 方式 2：直接使用（需手动关闭）
    milvus_manager.connect()
    try:
        collection = milvus_manager.get_collection()
        # ... 使用 collection ...
    finally:
        milvus_manager.close()
"""

# 日志记录器
from loguru import logger

# pymilvus: Milvus 的 Python 客户端库
# 包含连接管理、数据操作、Schema 定义等核心功能
from pymilvus import (
    Collection,          # ORM 风格的 Collection 对象
    CollectionSchema,   # Collection 的结构定义（字段列表、描述等）
    DataType,           # 数据类型枚举（VARCHAR、FLOAT_VECTOR、JSON 等）
    FieldSchema,        # 单个字段的定义
    MilvusClient,       # Milvus 的轻量级客户端（底层 API）
    connections,        # 连接管理（连接、断开、列出连接等）
    utility,           # 工具函数（检查 Collection 是否存在、删除 Collection 等）
    MilvusException,    # Milvus 操作异常基类
)

from app.config import config  # 全局配置，读取 Milvus 连接参数


def _patch_pymilvus_milvus_client_orm_alias() -> None:
    """修复 pymilvus ORM 别名兼容性问题

    问题背景：
    当使用 langchain-milvus 的 VectorStore 时，它内部会创建一个 MilvusClient，
    并将 MilvusClient._using 设置为类似 "cm-{id}" 的别名。
    但这个别名没有在 pymilvus.orm.connections 中注册。
    随后，如果代码使用 ORM 的 Collection(..., using="cm-{id}") 访问，
    会抛出 ConnectionNotExistException 异常。

    解决方案：
    通过 monkey patch（猴子补丁）的方式，在 MilvusClient.__init__ 执行后，
    强制将 _using 属性设置为 "default"（即 connections.connect 时使用的别名），
    从而与 ORM 的 Collection 操作保持一致。

    注意：
    - 这是针对 pymilvus 特定版本的兼容性修复
    - 使用 getattr 检查避免重复 patch（_done 标记）
    - 如果 pymilvus 版本变化导致接口不兼容，ImportError 会被静默忽略
    """
    # 使用模块属性作为标记，避免重复执行（单例模式的函数版）
    if getattr(_patch_pymilvus_milvus_client_orm_alias, "_done", False):
        return

    try:
        from pymilvus.milvus_client.milvus_client import MilvusClient
    except ImportError:
        # pymilvus 版本可能不包含此模块，静默返回
        return

    # 保存原始的 __init__ 方法引用
    _orig_init = MilvusClient.__init__

    def _wrapped_init(self, *args, **kwargs):
        """包装后的 __init__ 方法

        先执行原始初始化逻辑，然后将 _using 强制设为 "default"
        """
        _orig_init(self, *args, **kwargs)
        self._using = "default"

    # 应用 monkey patch
    MilvusClient.__init__ = _wrapped_init
    # 设置完成标记
    setattr(_patch_pymilvus_milvus_client_orm_alias, "_done", True)


class MilvusClientManager:
    """Milvus 客户端管理器（单例）

    封装 Milvus 连接、Collection 管理、资源清理等操作。
    提供启动时初始化、运行时访问、关闭时清理的完整生命周期管理。

    Collection Schema 设计：
    - id: 主键，VARCHAR 类型，最大 100 字符
    - vector: 向量字段，FLOAT_VECTOR 类型，1024 维（与 text-embedding-v4 对应）
    - content: 原始文本内容，VARCHAR 类型，最大 8000 字符
    - metadata: 元数据，JSON 类型（存储标题、来源、时间戳等）

    索引设计：
    - metric_type: L2（欧氏距离），适合文本嵌入向量的相似度计算
    - index_type: IVF_FLAT（倒排索引 + 精确检索），平衡速度和精度
    """

    # Collection 名称，用于标识业务知识库 Collection
    COLLECTION_NAME: str = "biz"

    # 向量维度，必须与 embedding 模型输出维度一致
    # text-embedding-v4 默认输出 1024 维向量
    # 如果维度不匹配，插入和检索会失败
    VECTOR_DIM: int = 1024

    # 主键字段最大长度（字符数）
    ID_MAX_LENGTH: int = 100

    # 文本内容字段最大长度（字符数）
    # 8000 字符约等于 4000 个中文字或 8000 个英文单词
    CONTENT_MAX_LENGTH: int = 8000

    # 默认分片数，用于数据分布和并行查询
    # 分片越多，可支持的并发查询越高，但管理复杂度也增加
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        # 私有字段，存储 MilvusClient 和 Collection 实例
        self._client: MilvusClient | None = None    # 低级客户端
        self._collection: Collection | None = None   # ORM Collection 对象

    def connect(self) -> MilvusClient:
        """连接到 Milvus 服务器并初始化 Collection

        这是应用启动时调用的入口方法，负责：
        1. 建立与 Milvus 服务器的连接
        2. 创建 MilvusClient 实例
        3. 检查并创建业务 Collection（如果不存在）
        4. 验证向量维度（不匹配时重建 Collection）
        5. 加载 Collection 到内存以支持查询

        Returns:
            MilvusClient: 连接好的客户端实例

        Raises:
            RuntimeError: 连接失败或 Collection 初始化失败时抛出
            MilvusException: Milvus 内部操作失败时抛出
            ConnectionError: 网络连接失败时抛出
        """
        # 幂等检查：如果已经连接，直接返回已有客户端
        # 避免重复初始化（应用生命周期中 connect 可能被调用多次）
        if self._collection is not None and self._client is not None:
            logger.debug("Milvus 已连接，跳过重复 connect")
            return self._client

        try:
            # 应用 pymilvus ORM 别名兼容性修复
            _patch_pymilvus_milvus_client_orm_alias()

            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 使用 connections 模块建立连接（ORM 风格）
            # alias="default" 是默认连接别名，与 langchain-milvus 兼容
            connections.connect(
                alias="default",                                  # 连接别名
                host=config.milvus_host,                          # Milvus 服务器地址
                port=str(config.milvus_port),                     # Milvus 端口（需要字符串）
                timeout=config.milvus_timeout / 1000,             # 超时时间（转换为秒）
            )

            # 创建 MilvusClient 实例（底层 API 风格）
            # MilvusClient 比 ORM Collection 更轻量，适合批量操作
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 检查 Collection 是否已存在
            if not self._collection_exists():
                # 不存在则创建
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                # 存在则获取引用，并检查向量维度
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)

                # 维度兼容性检查：确保 embedding 模型与存储的向量维度一致
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break

                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        # 维度不匹配，删除旧 Collection 并重建
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.COLLECTION_NAME}'...")
                        _ = utility.drop_collection(self.COLLECTION_NAME)
                        logger.info(f"正在重新创建 collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 将 Collection 加载到内存（Milvus 查询前必须加载）
            self._load_collection()

            return self._client

        except MilvusException as e:
            # pymilvus 抛出的业务异常（如 Collection 已存在、字段类型错误等）
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            # 网络连接失败
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            # 捕获所有未预期的异常
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 Collection 是否已存在于 Milvus 服务器

        Returns:
            bool: True 表示存在，False 表示不存在
        """
        # utility.has_collection 返回 bool（但类型标注可能不准确）
        result = utility.has_collection(self.COLLECTION_NAME)
        return bool(result)

    def _create_collection(self) -> None:
        """创建 biz Collection

        创建包含以下字段的 Collection：
        1. id: 主键，唯一标识每条记录
        2. vector: 1024 维浮点向量（文本嵌入）
        3. content: 原始文本内容
        4. metadata: JSON 格式的元数据（标题、来源等）

        并在创建后为 vector 字段建立 IVF_FLAT 索引。
        """
        # 定义 Collection 的字段列表
        fields = [
            # 主键字段：VARCHAR 类型，最大 100 字符
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,        # 可变长度字符串
                max_length=self.ID_MAX_LENGTH,  # 最大字符数
                is_primary=True,               # 标记为主键（唯一、必填）
            ),
            # 向量字段：存储文本嵌入后的高维向量
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,  # 32 位浮点向量
                dim=self.VECTOR_DIM,           # 向量维度（必须与 embedding 模型一致）
            ),
            # 文本内容字段：存储原始文本，便于检索后展示
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            # 元数字段：存储与向量相关的额外信息（JSON 格式）
            # 如：{"title": "第一章", "source": "user-guide.pdf", "page": 5}
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,  # JSON 类型，可存储任意结构化数据
            ),
        ]

        # 创建 Collection Schema（结构定义）
        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",  # Collection 描述信息
            enable_dynamic_field=False,  # 禁用动态字段（固定 Schema 更安全）
        )

        # 创建 Collection 实例
        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
            num_shards=self.DEFAULT_SHARD_NUMBER,  # 分片数，影响并行查询能力
        )

        # 创建向量字段索引（必须创建索引才能高效检索）
        self._create_index()

    def _create_index(self) -> None:
        """为 vector 字段创建索引

        索引是加速向量检索的关键数据结构。
        Milvus 支持多种索引类型：
        - IVF_FLAT: 倒排索引 + 精确检索，适合中等规模数据（<百万级）
        - HNSW: 分层可导航小世界图，高精度、高速度，但内存占用大
        - PQ: 产品量化，适合大规模数据压缩

        这里选择 IVF_FLAT 作为默认值，平衡速度和精度。

        metric_type=L2 表示使用欧氏距离计算相似度：
        - 欧氏距离越小，两个向量越相似
        - L2 适合文本嵌入场景
        - 另一种常用度量是 IP（内积），适合归一化向量
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化")

        # 索引参数
        index_params = {
            "metric_type": "L2",       # 欧氏距离（L2 距离）
            "index_type": "IVF_FLAT",  # 倒排索引 + 精确检索
            "params": {"nlist": 128},  # 聚类中心数量，影响索引精度和速度
        }

        # 创建索引（异步操作，Milvus 内部处理）
        _ = self._collection.create_index(
            field_name="vector",       # 要创建索引的字段
            index_params=index_params,  # 索引配置参数
        )

        logger.info("成功为 vector 字段创建索引")

    def _load_collection(self) -> None:
        """将 Collection 加载到内存

        Milvus 的工作流程：
        1. 数据写入磁盘（持久化存储）
        2. 查询前必须加载到内存（提高查询速度）
        3. 如果内存不足，Milvus 会使用内存映射（mmap）

        此方法兼容 Milvus 多个版本的 API：
        - 新版本：utility.load_state() 返回加载状态
        - 旧版本：直接调用 collection.load()，已加载会抛异常
        """
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        try:
            # 方法 1：尝试使用 utility.load_state（新版本 Milvus）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # 获取状态的字符串表示（兼容不同版本的返回类型）
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                # 未加载，执行加载操作
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2：直接尝试加载，捕获"已加载"异常（旧版本兼容）
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """获取 Collection 实例

        Returns:
            Collection: ORM Collection 对象

        Raises:
            RuntimeError: 如果尚未调用 connect() 初始化
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """执行健康检查

        检查 Milvus 连接是否正常。
        可被监控系统调用，用于判断服务可用性。

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            # 尝试列出连接（轻量级操作，不会触发实际查询）
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭 Milvus 连接并释放资源

        执行清理操作：
        1. 释放 Collection（从内存卸载）
        2. 断开与 Milvus 服务器的连接
        3. 清空客户端引用

        即使部分操作失败，也会继续执行其他清理步骤。
        错误信息会汇总后统一记录到日志。
        """
        errors = []

        # 步骤 1：释放 Collection（从内存卸载）
        try:
            if self._collection is not None:
                self._collection.release()  # 释放内存占用
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        # 步骤 2：断开连接
        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        # 步骤 3：清空客户端引用
        self._client = None

        # 步骤 4：汇总错误日志
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口（with 语句开始时调用）

        Returns:
            MilvusClientManager: 自身实例
        """
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出（with 语句结束时调用）

        无论是否发生异常，都会执行清理操作。

        Args:
            exc_type: 异常类型（如果有），None 表示正常退出
            exc_val: 异常实例（如果有）
            exc_tb: 异常回溯对象
        """
        self.close()


# ========== 全局 Milvus 客户端管理器单例 ==========
# 应用启动时通过 app.main:lifespan 中的 milvus_manager.connect() 初始化
# 应用关闭时通过 milvus_manager.close() 清理
milvus_manager = MilvusClientManager()
