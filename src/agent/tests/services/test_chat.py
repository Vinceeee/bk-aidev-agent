from unittest.mock import Mock, patch

import pytest
from aidev_agent.enums import PromptRole
from aidev_agent.exceptions import AgentException
from aidev_agent.services.chat import ChatCompletionAgent
from aidev_agent.services.pydantic_models import AgentOptions, ChatPrompt, ExecuteKwargs
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@pytest.fixture
def mock_chat_model():
    """创建模拟的聊天模型"""
    model = Mock(spec=BaseChatModel)
    model.model_name = "test-model"
    model.invoke.return_value = Mock(content="Test response", id="test-id", additional_kwargs={})
    return model


@pytest.fixture
def sample_chat_history():
    """创建示例聊天历史"""
    return [
        ChatPrompt(role="system", content="You are a helpful assistant"),
        ChatPrompt(role="user", content="Hello"),
        ChatPrompt(role="assistant", content="Hi there!"),
    ]


@pytest.fixture
def sample_agent_options():
    """创建示例智能体选项"""
    return AgentOptions()


class TestChatCompletionAgent:
    """测试 ChatCompletionAgent 类"""

    def test_init_with_required_params(self, mock_chat_model, sample_chat_history):
        """测试使用必需参数初始化"""
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        assert agent.chat_model == mock_chat_model
        assert agent.chat_history == sample_chat_history
        assert agent.files == []
        assert agent.tools is None
        assert agent.support_vision is False

    @pytest.mark.parametrize("support_vision,expected", [(True, True), (False, False)])
    def test_init_with_vision_support(self, mock_chat_model, sample_chat_history, support_vision, expected):
        """测试视觉支持参数"""
        agent = ChatCompletionAgent(
            chat_model=mock_chat_model, chat_history=sample_chat_history, support_vision=support_vision
        )

        assert agent.support_vision == expected

    def test_convert_history_to_messages(self, mock_chat_model, sample_chat_history):
        """测试聊天历史转换为消息"""
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        messages = agent.convert_history_to_messages()

        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert isinstance(messages[2], AIMessage)

    @pytest.mark.parametrize(
        "role,expected_type",
        [
            (PromptRole.USER.value, HumanMessage),
            (PromptRole.ASSISTANT.value, AIMessage),
            (PromptRole.AI.value, AIMessage),
            (PromptRole.SYSTEM.value, SystemMessage),
        ],
    )
    def test_chat_history_to_langchain_messages(self, mock_chat_model, role, expected_type):
        """测试不同角色的消息转换"""
        chat_history = [ChatPrompt(role=role, content="Test content")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history)

        messages = agent._chat_history_to_langchain_messages(chat_history)

        assert len(messages) == 1
        assert isinstance(messages[0], expected_type)

    def test_convert_contents_skip_guide_role(self, mock_chat_model):
        """测试跳过guide角色的内容"""
        chat_history = [
            ChatPrompt(role="guide", content="Guide content"),
            ChatPrompt(role="user", content="User content"),
        ]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history)

        converted = agent._convert_contents(chat_history)

        assert len(converted) == 1
        assert converted[0].role == "user"

    def test_convert_contents_hidden_to_user(self, mock_chat_model):
        """测试hidden角色转换为user"""
        chat_history = [ChatPrompt(role="hidden", content="Hidden content")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history)

        converted = agent._convert_contents(chat_history)

        assert len(converted) == 1
        assert converted[0].role == "user"

    def test_convert_contents_pause_to_assistant(self, mock_chat_model):
        """测试pause角色转换为assistant"""
        chat_history = [ChatPrompt(role="pause", content="Pause content")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history)

        converted = agent._convert_contents(chat_history)

        assert len(converted) == 1
        assert converted[0].role == "assistant"

    def test_convert_contents_user_image_without_vision_support(self, mock_chat_model):
        """测试不支持视觉时处理用户图片"""
        chat_history = [ChatPrompt(role="user-image", content="![image](http://example.com/image.jpg)")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history, support_vision=False)

        with pytest.raises(AgentException):
            agent._convert_contents(chat_history)

    def test_convert_contents_user_image_with_vision_support(self, mock_chat_model):
        """测试支持视觉时处理用户图片"""
        chat_history = [ChatPrompt(role="user-image", content="![image](http://example.com/image.jpg)")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history, support_vision=True)

        converted = agent._convert_contents(chat_history)

        assert len(converted) == 1
        assert converted[0].role == "user"
        assert "image.jpg" in converted[0].content
        assert len(agent.files) == 1

    def test_convert_contents_deepseek_r1_system_to_user(self, mock_chat_model):
        """测试deepseek-r1模型将system转换为user"""
        mock_chat_model.model_name = "deepseek-r1-test"
        chat_history = [ChatPrompt(role="system", content="System prompt")]
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=chat_history)

        converted = agent._convert_contents(chat_history)

        assert len(converted) == 1
        assert converted[0].role == "user"

    def test_model_name_property(self, mock_chat_model, sample_chat_history):
        """测试model_name属性"""
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        assert agent.model_name == "test-model"

    @pytest.mark.parametrize(
        "has_tools,has_files,has_knowledge,expected",
        [
            (True, False, False, True),
            (False, True, False, True),
            (False, False, True, True),
            (False, False, False, False),
        ],
    )
    def test_is_run_by_agent(self, mock_chat_model, sample_chat_history, has_tools, has_files, has_knowledge, expected):
        """测试是否需要通过agent运行"""
        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        # 直接设置属性，避免Pydantic验证问题
        if has_tools:
            agent.tools = [Mock()]
        if has_files:
            agent.files = [{"name": "test.txt"}]
        if has_knowledge:
            agent.knowledge_bases = [{"id": 1}]

        assert agent.is_run_by_agent() == expected

    def test_execute_without_agent(self, mock_chat_model, sample_chat_history):
        """测试不通过agent执行"""
        # 创建一个正确的 AIMessage 返回值
        from langchain_core.messages import AIMessage

        mock_response = AIMessage(content="Test response", id="test-id")
        mock_chat_model.invoke.return_value = mock_response

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        execute_kwargs = ExecuteKwargs(stream=False, run_agent=False)
        result = agent.execute(execute_kwargs)

        assert "choices" in result
        assert "model" in result
        mock_chat_model.invoke.assert_called_once()

    @patch("aidev_agent.services.chat.ChatCompletionAgent._get_agent")
    def test_execute_with_agent_non_stream(self, mock_get_agent, mock_chat_model, sample_chat_history):
        """测试通过agent执行（非流式）"""
        mock_agent_executor = Mock()
        mock_config = Mock()
        mock_get_agent.return_value = (mock_agent_executor, mock_config)

        # 模拟异步调用结果
        with patch("aidev_agent.services.chat.get_event_loop") as mock_loop:
            mock_loop.return_value.run_until_complete.return_value = {"output": "Test response"}

            agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)
            # 直接设置工具属性以触发agent模式
            agent.tools = [Mock()]

            execute_kwargs = ExecuteKwargs(stream=False, run_agent=True)
            result = agent.execute(execute_kwargs)

            assert result == "Test response"

    def test_stream_normal_content(self, mock_chat_model, sample_chat_history):
        """测试流式处理正常内容"""
        # 模拟流式响应
        mock_chunks = [
            Mock(content="Hello", additional_kwargs={}),
            Mock(content=" world", additional_kwargs={}),
            Mock(content="!", additional_kwargs={}),
        ]
        mock_chat_model.stream.return_value = iter(mock_chunks)

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        messages = agent.convert_history_to_messages()
        result_generator = agent._stream(messages)

        # 收集所有结果
        results = list(result_generator)

        # 验证结果格式
        assert len(results) > 0
        assert results[-1] == "data: [DONE]\n\n"

    def test_stream_with_reasoning_content(self, mock_chat_model, sample_chat_history):
        """测试流式处理包含推理内容"""
        mock_chunks = [
            Mock(content="", additional_kwargs={"reasoning_content": "Thinking..."}),
            Mock(content="Answer", additional_kwargs={}),
        ]
        mock_chat_model.stream.return_value = iter(mock_chunks)

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        messages = agent.convert_history_to_messages()
        result_generator = agent._stream(messages)

        results = list(result_generator)

        # 验证包含推理和文本事件
        data_lines = [line for line in results if line.startswith("data: ") and line != "data: [DONE]\n\n"]
        assert len(data_lines) > 0

    def test_stream_exception_handling(self, mock_chat_model, sample_chat_history):
        """测试流式处理异常情况"""
        mock_chat_model.stream.side_effect = Exception("Stream error")

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        messages = agent.convert_history_to_messages()
        result_generator = agent._stream(messages)

        results = list(result_generator)

        # 验证包含错误信息
        error_found = False
        for result in results:
            if "error" in result:
                error_found = True
                break

        assert error_found

    def test_invoke_non_stream(self, mock_chat_model, sample_chat_history):
        """测试非流式调用"""
        # 创建一个正确的 AIMessage 返回值
        from langchain_core.messages import AIMessage

        mock_response = AIMessage(content="Test response", id="test-id")
        mock_chat_model.invoke.return_value = mock_response

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        messages = agent.convert_history_to_messages()
        result = agent._invoke(messages)

        assert "choices" in result
        assert "model" in result
        assert "id" in result
        mock_chat_model.invoke.assert_called_once_with(input=messages)

    @patch("aidev_agent.services.chat.ConversationTokenBufferMemory")
    def test_get_memory_window(self, mock_memory_class, mock_chat_model, sample_chat_history):
        """测试获取内存窗口"""
        mock_memory = Mock()
        mock_memory.buffer = sample_chat_history[:2]  # 模拟缓冲区
        mock_memory_class.return_value = mock_memory

        agent = ChatCompletionAgent(chat_model=mock_chat_model, chat_history=sample_chat_history)

        window_size = agent.get_memory_window(max_token_limit=4096)

        # 验证返回值为缓冲区长度与聊天历史长度的差值
        expected = len(mock_memory.buffer) - len(sample_chat_history)
        assert window_size == expected
