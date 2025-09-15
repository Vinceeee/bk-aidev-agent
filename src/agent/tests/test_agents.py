import unittest
from unittest.mock import MagicMock

from aidev_agent.core.agent.agents import (
    EnhancedJSONAgentOutputParser,
    create_enhanced_structured_chat_agent,
    get_beijing_now,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool


class TestAgents(unittest.TestCase):
    def setUp(self):
        self.llm = MagicMock(spec=BaseChatModel)
        self.tools = [MagicMock(spec=BaseTool)]

    def test_get_beijing_now(self):
        """测试获取北京时间函数。"""
        result = get_beijing_now()
        self.assertIsInstance(result, str)
        self.assertIn("年", result)
        self.assertIn("月", result)
        self.assertIn("日", result)

    def test_EnhancedJSONAgentOutputParser(self):
        """测试EnhancedJSONAgentOutputParser的解析功能。"""
        parser = EnhancedJSONAgentOutputParser(self.llm)
        test_text = '{"action": "Final Answer", "action_input": "test output"}'
        result = parser.parse(test_text)
        self.assertEqual(result.return_values["output"], "test output")

    def test_create_enhanced_structured_chat_agent(self):
        """测试创建增强型结构化聊天代理。"""
        agent = create_enhanced_structured_chat_agent(self.llm, self.tools, MagicMock())
        self.assertIsNotNone(agent)
