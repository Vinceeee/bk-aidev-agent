from typing import List
from unittest.mock import Mock, patch

import pytest
from aidev_agent.api.abstract_client import AbstractBKAidevResourceManager
from aidev_agent.enums import AgentBuildType, AgentType
from aidev_agent.services.agent import AgentInstanceFactory
from aidev_agent.services.chat import ChatCompletionAgent
from aidev_agent.services.config_manager import AgentConfig
from aidev_agent.services.pydantic_models import AgentOptions


class MockResourceManager(AbstractBKAidevResourceManager):
    """Mock resource manager for testing"""

    def __init__(self, session_context_data=None, agent_config_data=None):
        self.session_context_data = session_context_data or []
        self.agent_config_data = agent_config_data or {}

    def get_chat_session_context(self, session_code: str, **kwargs) -> List[dict]:
        return self.session_context_data

    def retrieve_knowledgebase(self, id: int, **kwargs) -> dict:
        return {"id": id, "name": f"KB_{id}"}

    def retrieve_knowledge(self, id: int, **kwargs) -> dict:
        return {"id": id, "name": f"Knowledge_{id}"}

    def construct_tool(self, tool_code: str, **kwargs):
        return Mock(name=f"tool_{tool_code}")

    def retrieve_agent_config(self, agent_code: str, **kwargs) -> dict:
        return self.agent_config_data.get(agent_code, {})

    def knowledge_query(self, data: dict) -> dict:
        return {}


@pytest.fixture
def mock_resource_manager():
    return MockResourceManager()


@pytest.fixture
def sample_agent_config():
    return AgentConfig(
        agent_code="test_agent",
        agent_name="Test Agent",
        llm_model_name="test-model",
        non_thinking_llm_model_name="test-non-thinking",
        role_prompt="You are a test agent",
        knowledgebase_ids=[1, 2],
        knowledge_ids=[3, 4],
        tool_codes=["tool1", "tool2"],
        opening_mark="Hello!",
        generating_keyword="生成中",
        mcp_server_config=None,
        agent_options=AgentOptions(),
        command_agent_mapping={},
    )


class TestAgentInstanceFactory:
    """测试 AgentInstanceFactory 类"""

    def test_init_with_defaults(self):
        """测试默认参数初始化"""
        factory = AgentInstanceFactory("test_agent")

        assert factory.agent_code == "test_agent"
        assert factory.agent_type == AgentType.CHAT
        assert factory.build_type == AgentBuildType.SESSION
        assert factory.session_code is None
        assert factory.callbacks == []

    @pytest.mark.parametrize(
        "agent_type,build_type",
        [
            (AgentType.CHAT, AgentBuildType.SESSION),
            (AgentType.CHAT, AgentBuildType.DIRECT),
        ],
    )
    def test_init_with_parameters(self, agent_type, build_type):
        """测试参数化初始化"""
        factory = AgentInstanceFactory(
            agent_code="test_agent",
            agent_type=agent_type,
            build_type=build_type,
            session_code="test_session" if build_type == AgentBuildType.SESSION else None,
        )

        assert factory.agent_type == agent_type
        assert factory.build_type == build_type

    def test_validate_params_session_without_code(self):
        """测试会话模式缺少session_code的验证"""
        factory = AgentInstanceFactory("test_agent", build_type=AgentBuildType.SESSION)

        with pytest.raises(ValueError, match="session_code is required"):
            factory._validate_params()

    def test_validate_params_unsupported_build_type(self):
        """测试不支持的构建类型"""
        factory = AgentInstanceFactory("test_agent")
        factory.build_type = "unsupported"

        with pytest.raises(ValueError, match="Unsupported build_type"):
            factory._validate_params()

    @patch("aidev_agent.services.agent.AgentConfigManager.get_config")
    def test_build_from_session(self, mock_get_config, mock_resource_manager, sample_agent_config):
        """测试从会话构建基础参数"""
        mock_get_config.return_value = sample_agent_config
        mock_resource_manager.session_context_data = [{"role": "user", "content": "Hello"}]

        factory = AgentInstanceFactory(
            "test_agent", session_code="test_session", resource_manager=mock_resource_manager
        )

        result = factory._build_from_session()

        assert result["agent_code"] == "test_agent"
        assert len(result["session_context_data"]) == 1
        assert result["switch_agent"] is False

    def test_build_direct(self):
        """测试直接构建基础参数"""
        session_data = [{"role": "user", "content": "Hello"}]
        factory = AgentInstanceFactory("test_agent")

        result = factory._build_direct(session_data)

        assert result["agent_code"] == "test_agent"
        assert result["session_context_data"] == session_data
        assert result["switch_agent"] is False

    @patch("aidev_agent.services.agent.ChatModel.get_setup_instance")
    @patch("aidev_agent.services.agent.AgentConfigManager.get_config")
    def test_build_chat_model(self, mock_get_config, mock_chat_model, sample_agent_config):
        """测试构建聊天模型"""
        mock_get_config.return_value = sample_agent_config
        mock_chat_model.return_value = Mock()

        factory = AgentInstanceFactory("test_agent")
        result = factory.build_chat_model("test_agent")

        mock_chat_model.assert_called_once()
        assert result is not None

    def test_build_chat_history(self):
        """测试构建聊天历史"""
        factory = AgentInstanceFactory("test_agent")
        session_data = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

        result = factory.build_chat_history(session_data)

        assert len(result) == 2
        assert all(hasattr(item, "role") and hasattr(item, "content") for item in result)

    @patch("aidev_agent.services.agent.AgentConfigManager.get_config")
    def test_build_knowledge_bases(self, mock_get_config, mock_resource_manager, sample_agent_config):
        """测试构建知识库"""
        mock_get_config.return_value = sample_agent_config

        factory = AgentInstanceFactory("test_agent", resource_manager=mock_resource_manager)
        result = factory.build_knowledge_bases("test_agent")

        assert len(result) == 2
        assert all("id" in kb for kb in result)

    @patch("aidev_agent.services.agent.AgentConfigManager.get_config")
    def test_build_tools(self, mock_get_config, mock_resource_manager, sample_agent_config):
        """测试构建工具"""
        mock_get_config.return_value = sample_agent_config

        factory = AgentInstanceFactory("test_agent", resource_manager=mock_resource_manager)
        result = factory.build_tools("test_agent")

        assert len(result) == 2

    def test_check_agent_switch_with_command(self, sample_agent_config):
        """测试命令触发的智能体切换"""
        sample_agent_config.command_agent_mapping = {"translate": "translator_agent"}
        session_data = [{"role": "user", "content": "Translate this", "extra": {"command": "translate"}}]

        factory = AgentInstanceFactory("test_agent")
        switch_agent, final_agent_code = factory._check_agent_switch(session_data, sample_agent_config)

        assert switch_agent is True
        assert final_agent_code == "translator_agent"

    def test_check_agent_switch_no_command(self, sample_agent_config):
        """测试无命令时不切换智能体"""
        session_data = [{"role": "user", "content": "Hello"}]

        factory = AgentInstanceFactory("test_agent")
        switch_agent, final_agent_code = factory._check_agent_switch(session_data, sample_agent_config)

        assert switch_agent is False
        assert final_agent_code == "test_agent"

    def test_clean_last_assistant_message_with_keyword(self, sample_agent_config):
        """测试清理包含生成关键词的最后一条assistant消息"""
        session_data = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "生成中..."}]

        factory = AgentInstanceFactory("test_agent", session_code="test")
        factory._clean_last_assistant_message(session_data, sample_agent_config)

        assert len(session_data) == 1
        assert session_data[-1]["role"] == "user"

    def test_clean_last_assistant_message_without_keyword(self, sample_agent_config):
        """测试不清理不包含生成关键词的assistant消息"""
        session_data = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
        original_length = len(session_data)

        factory = AgentInstanceFactory("test_agent", session_code="test")
        factory._clean_last_assistant_message(session_data, sample_agent_config)

        assert len(session_data) == original_length

    def test_register_agent_type(self):
        """测试注册新的智能体类型"""
        # 清理现有注册
        original_classes = AgentInstanceFactory._agent_classes.copy()
        original_builders = AgentInstanceFactory._agent_builders.copy()

        try:
            # 注册新类型
            AgentInstanceFactory.register_agent_type(
                AgentType.CHAT, ChatCompletionAgent, AgentInstanceFactory.build_chat_agent_args, override=True
            )

            assert AgentType.CHAT in AgentInstanceFactory._agent_classes
            assert AgentType.CHAT in AgentInstanceFactory._agent_builders
        finally:
            # 恢复原始状态
            AgentInstanceFactory._agent_classes = original_classes
            AgentInstanceFactory._agent_builders = original_builders

    def test_register_agent_type_duplicate_without_override(self):
        """测试重复注册智能体类型但不覆盖"""
        # 确保CHAT类型已注册
        AgentInstanceFactory._agent_classes[AgentType.CHAT] = ChatCompletionAgent

        with pytest.raises(ValueError, match="already exists"):
            AgentInstanceFactory.register_agent_type(
                AgentType.CHAT, ChatCompletionAgent, AgentInstanceFactory.build_chat_agent_args, override=False
            )
