# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""

from collections import deque
from unittest.mock import Mock, patch

import pytest
from aidev_agent.core.extend.agent.qa import (
    CommonQAAgent,
    CommonQAStreamingMixIn,
    IntentRecognitionMixin,
    StructuredChatCommonQAAgent,
    ToolCallingCommonQAAgent,
)
from aidev_agent.enums import ContextType, EventType
from aidev_agent.services.pydantic_models import AgentOptions
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool


@pytest.fixture
def mock_llm():
    """创建模拟的LLM"""
    llm = Mock(spec=BaseChatModel)
    llm.model_name = "test-model"
    llm.get_num_tokens_from_messages.return_value = 100
    return llm


@pytest.fixture
def mock_tools():
    """创建模拟的工具列表"""
    tool1 = Mock(spec=BaseTool)
    tool1.name = "test_tool_1"
    tool2 = Mock(spec=BaseTool)
    tool2.name = "test_tool_2"
    return [tool1, tool2]


@pytest.fixture
def sample_agent_options():
    """创建示例智能体选项"""
    return AgentOptions()


@pytest.fixture
def sample_intermediate_steps():
    """创建示例中间步骤"""
    action = AgentAction(tool="test_tool", tool_input={"param": "value"}, log="test log")
    return [(action, "test result")]


@pytest.fixture
def sample_kwargs():
    """创建示例kwargs"""
    return {
        "input": "test query",
        "query": "test query",
        "chat_history": [HumanMessage(content="Hello")],
        "context": ["test context"],
        "qa_context": ["test qa context"],
        "context_type": ContextType.PRIVATE.value,
        "role_prompt": "You are a helpful assistant",
        "use_general_knowledge_on_miss": True,
        "rejection_response": "Sorry, I cannot help with that",
        "enable_parallel_tool_calls": True,
    }


class TestIntentRecognitionMixin:
    """测试IntentRecognitionMixin类"""

    def test_init_subclass(self):
        """测试子类初始化时模板合并"""

        class TestMixin(IntentRecognitionMixin):
            qa_prompt_templates = {"test_template": "test_value"}

        assert "test_template" in TestMixin.qa_prompt_templates

    @pytest.mark.parametrize(
        "doc_type,expected_content",
        [
            ({"metadata": {"index_content": "index", "knowledge_base_id": "kb1"}, "page_content": "page"}, "index"),
            ({"metadata": {"knowledge_base_id": "kb1"}, "page_content": "page"}, "page"),
        ],
    )
    def test_knowledge_resources_postproc(self, doc_type, expected_content):
        """测试知识资源后处理"""
        kwargs = {}
        recog_results = {
            "knowledge_resources_highly_relevant": [doc_type],
            "qa_response_kb_ids": [],
        }

        with patch("aidev_agent.core.extend.agent.qa.is_structured_data", return_value=True):
            with patch("aidev_agent.core.extend.agent.qa.conditional_dispatch_custom_event"):
                with patch("aidev_agent.core.extend.agent.qa.deduplicate_knowledge_file_paths", return_value=[]):
                    IntentRecognitionMixin.knowledge_resources_postproc(
                        kwargs, recog_results, "knowledge_resources_highly_relevant"
                    )

        assert kwargs["context"] == [expected_content]

    @pytest.mark.parametrize(
        "header,rows,expected_lines",
        [
            (["Name", "ID"], [["Tool1", "1"], ["Tool2", "2"]], 4),  # header + separator + 2 rows
        ],
    )
    def test_pretty_table(self, header, rows, expected_lines):
        """测试表格格式化"""
        result = IntentRecognitionMixin.pretty_table(header, rows)
        lines = result.split("\n")
        assert len(lines) == expected_lines
        assert "Name" in lines[0]
        assert "ID" in lines[0]

    def test_cell_display_length(self):
        """测试字符显示长度计算"""
        # 测试英文字符
        assert IntentRecognitionMixin.cell_display_length("hello") == 5
        # 测试中文字符
        assert IntentRecognitionMixin.cell_display_length("你好") == 4

    @pytest.mark.parametrize(
        "intents,category,expected",
        [
            ([{"资源类别": "knowledge base", "资源ID": "1"}], "knowledge base", [1]),
            ([{"资源类别": "knowledge item", "资源ID": "2"}], "knowledge item", [2]),
        ],
    )
    def test_get_intent_ids(self, intents, category, expected):
        """测试按类别收集资源ID"""
        from aidev_agent.enums import IntentCategory

        result = IntentRecognitionMixin.get_intent_ids(intents, IntentCategory(category))
        assert result == expected

    def test_get_intent_ids_with_tool_fails(self):
        """测试工具类型的资源ID转换会失败（因为原方法有bug）"""
        from aidev_agent.enums import IntentCategory

        intents = [{"资源类别": "tool", "资源ID": "test_tool"}]

        # 这个测试验证了原方法的一个bug：工具ID不应该转换为int
        with pytest.raises(ValueError, match="invalid literal for int"):
            IntentRecognitionMixin.get_intent_ids(intents, IntentCategory.TOOL)

    @pytest.mark.parametrize(
        "title,data_list,empty_msg,expected_contains",
        [
            ("测试标题", [["类别1", "ID1"]], "为空", "测试标题"),
            ("空标题", [], "无数据", "无数据"),
        ],
    )
    def test_build_table(self, title, data_list, empty_msg, expected_contains):
        """测试构建表格内容"""
        result = IntentRecognitionMixin.build_table(title, data_list, empty_msg)
        assert expected_contains in result


class TestCommonQAStreamingMixIn:
    """测试CommonQAStreamingMixIn类"""

    def test_common_filter_with_hit(self):
        """测试过滤器命中情况"""
        mixin = CommonQAStreamingMixIn()
        cache = deque([{"event": EventType.THINK.value, "content": "<think>test content</think>", "cover": False}])

        hit, recall_event = mixin.common_filter(cache, ["<think>", "</think>"], EventType.THINK.value)

        assert hit is True
        # 根据实际实现，只会替换第一个匹配的符号
        assert recall_event["content"] == "test content</think>"

    def test_common_filter_no_hit(self):
        """测试过滤器未命中情况"""
        mixin = CommonQAStreamingMixIn()
        cache = deque([{"event": EventType.TEXT.value, "content": "normal content", "cover": False}])

        hit, recall_event = mixin.common_filter(cache, ["<think>"], EventType.THINK.value)

        assert hit is False
        assert recall_event is None

    def test_cache_filter(self):
        """测试缓存过滤"""
        mixin = CommonQAStreamingMixIn()
        cache = deque(
            [
                {"event": EventType.THINK.value, "content": "<think>thinking</think>", "cover": False},
                {"event": EventType.TEXT.value, "content": "final answer", "cover": False},
            ]
        )

        filtered_cache = mixin.cache_filter(cache)

        # 验证过滤后的结果
        assert len(filtered_cache) >= 1
        # 验证TEXT事件保持不变
        text_events = [item for item in filtered_cache if item["event"] == EventType.TEXT.value]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "final answer"

    def test_check_and_append(self):
        """测试检查和追加逻辑"""
        mixin = CommonQAStreamingMixIn()
        cache = deque([{"event": EventType.THINK.value, "content": "thinking", "cover": False}])
        ret = {"event": EventType.TEXT.value, "content": "answer", "cover": False}

        mixin.check_and_append(cache, ret)

        # 验证在think和text之间添加了换行
        assert ret["content"] == "\n\nanswer"
        assert len(cache) == 2

    def test_yield_ret(self):
        """测试返回格式化"""
        mixin = CommonQAStreamingMixIn()

        # 测试普通返回
        result = mixin._yield_ret({"event": "test", "content": "data"})
        assert result.startswith("data: ")
        assert "test" in result

        # 测试完成返回
        result = mixin._yield_ret(done=True)
        assert result == "data: [DONE]\n\n"

    def test_extract_exception_msg(self):
        """测试异常消息提取"""
        mixin = CommonQAStreamingMixIn()

        # 测试普通异常
        exception = Exception("test error")
        result = mixin._extract_exception_msg(exception)
        assert "test error" in result

        # 测试带response_data方法的异常
        exception = Mock()
        exception.response_data.return_value = "custom error"
        result = mixin._extract_exception_msg(exception)
        assert result == "custom error"


class TestToolCallingCommonQAAgent:
    """测试ToolCallingCommonQAAgent类"""

    def test_inheritance(self):
        """测试类继承关系"""
        # 测试MRO（方法解析顺序）而不是实例化
        mro = ToolCallingCommonQAAgent.__mro__
        class_names = [cls.__name__ for cls in mro]

        assert "IntentRecognitionMixin" in class_names
        assert "CommonQAStreamingMixIn" in class_names


class TestStructuredChatCommonQAAgent:
    """测试StructuredChatCommonQAAgent类"""

    def test_inheritance(self):
        """测试类继承关系"""
        # 测试MRO（方法解析顺序）而不是实例化
        mro = StructuredChatCommonQAAgent.__mro__
        class_names = [cls.__name__ for cls in mro]

        assert "IntentRecognitionMixin" in class_names
        assert "CommonQAStreamingMixIn" in class_names


class TestCommonQAAgent:
    """测试CommonQAAgent类"""

    def test_register_agent_class(self):
        """测试注册智能体类"""
        original_classes = CommonQAAgent.agent_classes.copy()

        try:
            CommonQAAgent.register_agent_class("test_agent", Mock)
            assert "test_agent" in CommonQAAgent.agent_classes
        finally:
            CommonQAAgent.agent_classes = original_classes

    @pytest.mark.parametrize(
        "has_function_calling,has_tools,expected_class",
        [
            (False, True, "structured_chat_common_qa_agent"),
            (True, True, "tool_calling_common_qa_agent"),
            (True, False, "tool_calling_common_qa_agent"),
        ],
    )
    @patch("aidev_agent.core.extend.agent.qa.is_model_without_function_calling")
    def test_get_agent_executor(
        self, mock_is_model_without_fc, has_function_calling, has_tools, expected_class, mock_llm, sample_agent_options
    ):
        """测试获取智能体执行器"""
        mock_is_model_without_fc.return_value = not has_function_calling

        extra_tools = [Mock()] if has_tools else []

        with patch.object(CommonQAAgent.agent_classes[expected_class], "get_agent_executor") as mock_get_executor:
            mock_get_executor.return_value = Mock()

            result = CommonQAAgent.get_agent_executor(
                llm=mock_llm, extra_tools=extra_tools, agent_options=sample_agent_options
            )

            mock_get_executor.assert_called_once()
            assert result is not None
