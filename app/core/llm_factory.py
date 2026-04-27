"""LLM 工厂模块

使用 LangChain ChatOpenAI 通过 OpenAI 兼容模式调用阿里云 DashScope API。
这种设计使得项目不依赖特定模型提供商，可以轻松切换到其他支持 OpenAI API 兼容格式的服务商。

支持的模型提供商（只需修改 base_url 和 api_key）：
- 阿里云 DashScope: https://dashscope.aliyuncs.com/compatible-mode/v1
- OpenAI 官方: https://api.openai.com/v1
- Azure OpenAI: https://{resource}.openai.azure.com
- 其他兼容 OpenAI API 格式的自建模型服务

设计模式：
- 工厂模式（Factory Pattern）：将 LLM 实例的创建逻辑封装在工厂类中
- 配置与代码分离：模型参数通过 config 读取，便于运行时切换
- 单例模式：llm_factory 是全局共享实例，避免重复创建
"""

from langchain_openai import ChatOpenAI  # LangChain 的 OpenAI 兼容聊天模型客户端
from app.config import config           # 全局配置，读取模型名称、API 密钥等
from loguru import logger               # 项目日志记录器


class LLMFactory:
    """LLM（大语言模型）工厂类

    负责创建标准化配置的 ChatOpenAI 实例。
    所有需要调用 LLM 的地方都应该通过此类创建实例，而不是直接实例化 ChatOpenAI。

    这样做的好处：
    1. 集中管理 API 端点、模型名称等配置
    2. 统一设置 temperature、streaming 等默认参数
    3. 未来切换模型提供商时只需修改此文件

    使用示例：
        # 创建使用默认配置的模型
        llm = llm_factory.create_chat_model()

        # 创建自定义温度的模型（用于创意写作）
        creative_llm = llm_factory.create_chat_model(temperature=1.0)

        # 创建非流式响应的模型（用于需要一次性获取完整结果的场景）
        batch_llm = llm_factory.create_chat_model(streaming=False)
    """

    # 阿里云 DashScope 的 OpenAI 兼容模式 API 地址
    # DashScope 提供了与 OpenAI API 兼容的接口格式
    # 文档参考：https://help.aliyun.com/zh/model-studio/getting-started/models
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @staticmethod
    def create_chat_model(
        model: str | None = None,
        temperature: float = 0.7,
        streaming: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> ChatOpenAI:
        """创建 ChatOpenAI 实例

        创建配置好的大语言模型客户端实例。
        所有参数都有默认值，可以只传需要的参数进行覆盖。

        Args:
            model: 模型名称，默认为 config.dashscope_model（qwen-max）
                  可选值参考 DashScope 文档：
                  - qwen-max: 效果最好，成本最高
                  - qwen-plus: 性价比均衡
                  - qwq-32b: 带推理能力（思考模型）
            temperature: 温度参数，控制输出的随机性（0.0-2.0）
                        0.7 是平衡默认值：
                        - 越低（接近 0）：输出越确定、保守
                        - 越高（接近 2）：输出越随机、有创意
            streaming: 是否启用流式输出
                       True：边生成边返回（响应快，用户体验好）
                       False：等全部生成后一次性返回（便于后处理）
            base_url: API 端点地址，默认为 DashScope 地址
                      可切换为其他 OpenAI 兼容服务
            api_key: API 密钥，默认为 config.dashscope_api_key
                     实际使用时必须配置有效的密钥，否则调用会失败

        Returns:
            ChatOpenAI: 配置好的 LangChain 模型实例
                        可以直接调用 .invoke() 进行对话

        Raises:
            AuthenticationError: API 密钥无效时
            RateLimitError: 请求频率超限时
            APIStatusError: API 返回错误状态码时
        """
        # 使用传入的值或默认值（惰性赋值）
        # or 运算符在值为 None 或空字符串时使用默认值
        model = model or config.dashscope_model
        base_url = base_url or LLMFactory.DASHSCOPE_BASE_URL
        api_key = api_key or config.dashscope_api_key

        # extra_body 用于传递 DashScope 特有的参数
        # stream 参数确保 DashScope 返回流式响应格式
        extra_body = {}
        extra_body["stream"] = streaming

        # 创建 ChatOpenAI 实例
        # LangChain 的 ChatOpenAI 会自动处理：
        # 1. HTTP 请求的发送和接收
        # 2. 请求体的构建（遵循 OpenAI Chat Completions API 格式）
        # 3. 响应解析（将 OpenAI 格式响应转换为 LangChain 标准格式）
        # 4. 重试逻辑和错误处理
        llm = ChatOpenAI(
            model=model,               # 模型名称
            temperature=temperature,    # 随机性温度
            streaming=streaming,        # 是否流式输出
            base_url=base_url,         # API 端点
            api_key=api_key,           # 认证密钥
            # extra_body 传递供应商特定参数，仅在有参数时传入
            extra_body=extra_body if extra_body else None,
        )

        return llm


# ========== 全局 LLM 工厂单例 ==========
# 模块加载时创建全局实例，整个应用共享
# 注意：这只是工厂实例本身，每次调用 create_chat_model() 仍会创建新的 LLM 实例
# 这样做是为了支持不同配置的 LLM（如 RAG 用快速模型，AIOps 用强模型）
llm_factory = LLMFactory()
