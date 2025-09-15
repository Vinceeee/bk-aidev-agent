from unittest.mock import Mock, patch

import pytest
from aidev_agent.services.command_handler import (
    CommandHandler,
    CommandProcessor,
    CommonCommandHandler,
    CommonCommandHandlerBuilder,
    ExplanationCommandHandler,
    TranslateCommandHandler,
)


class TestCommandHandler:
    """测试 CommandHandler 基类"""

    def test_abstract_get_template(self):
        """测试抽象方法get_template"""
        # CommandHandler是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            CommandHandler()

    def test_extract_context_vars_basic(self):
        """测试基本的上下文变量提取"""

        # 创建一个具体的子类用于测试
        class TestHandler(CommandHandler):
            def get_template(self) -> str:
                return "test template"

        handler = TestHandler()
        context = [
            {"__key": "name", "__value": "John", "other": "data"},
            {"__key": "age", "__value": "25", "context_type": "input"},
        ]

        variables = handler.extract_context_vars(context)

        assert variables == {"name": "John", "age": "25"}

    def test_extract_context_vars_empty(self):
        """测试空上下文的变量提取"""

        class TestHandler(CommandHandler):
            def get_template(self) -> str:
                return "test template"

        handler = TestHandler()
        variables = handler.extract_context_vars([])

        assert variables == {}

    def test_extract_context_vars_missing_keys(self):
        """测试缺少必要键的上下文"""

        class TestHandler(CommandHandler):
            def get_template(self) -> str:
                return "test template"

        handler = TestHandler()
        context = [{"__key": "name", "missing_value": "John"}, {"__value": "25", "missing_key": "age"}]

        variables = handler.extract_context_vars(context)

        assert variables == {}

    @patch("aidev_agent.services.command_handler.SafeJinjaEnvironment")
    def test_process_content(self, mock_jinja_env):
        """测试内容处理"""
        # 模拟Jinja环境
        mock_env_instance = Mock()
        mock_env_instance.render.return_value = "Rendered content"
        mock_jinja_env.return_value = mock_env_instance

        class TestHandler(CommandHandler):
            def get_template(self) -> str:
                return "Hello {{ name }}"

        handler = TestHandler()
        context = [{"__key": "name", "__value": "World"}]

        result = handler.process_content(context)

        assert result == "Rendered content"
        mock_env_instance.render.assert_called_once_with("Hello {{ name }}", {"name": "World"})


class TestTranslateCommandHandler:
    """测试翻译命令处理器"""

    def test_command_attribute(self):
        """测试命令属性"""
        handler = TranslateCommandHandler()
        assert handler.command == "translate"

    def test_get_template(self):
        """测试获取模板"""
        handler = TranslateCommandHandler()
        template = handler.get_template()

        assert "{{ language }}" in template
        assert "{{ content }}" in template
        assert "翻译" in template

    @pytest.mark.parametrize(
        "context,expected_vars",
        [
            (
                [{"__key": "content", "__value": "Hello"}, {"__key": "language", "__value": "中文"}],
                {"content": "Hello", "language": "中文"},
            ),
            (
                [{"__key": "content", "__value": "Bonjour"}, {"__key": "language", "__value": "English"}],
                {"content": "Bonjour", "language": "English"},
            ),
        ],
    )
    def test_extract_context_vars_valid(self, context, expected_vars):
        """测试有效的上下文变量提取"""
        handler = TranslateCommandHandler()
        variables = handler.extract_context_vars(context)

        assert variables == expected_vars

    def test_extract_context_vars_missing_content(self):
        """测试缺少content的情况"""
        handler = TranslateCommandHandler()
        context = [{"__key": "language", "__value": "中文"}]

        with pytest.raises(ValueError, match="Translation requires 'content'"):
            handler.extract_context_vars(context)

    @patch("aidev_agent.services.command_handler.SafeJinjaEnvironment")
    def test_process_content_complete(self, mock_jinja_env):
        """测试完整的内容处理流程"""
        mock_env_instance = Mock()
        mock_env_instance.render.return_value = (
            "请将以下内容翻译为中文:\nHello\n翻译要求: 确保翻译准确无误，无需冗余回答内容"
        )
        mock_jinja_env.return_value = mock_env_instance

        handler = TranslateCommandHandler()
        context = [{"__key": "content", "__value": "Hello"}, {"__key": "language", "__value": "中文"}]

        result = handler.process_content(context)

        assert "翻译" in result
        assert "Hello" in result


class TestExplanationCommandHandler:
    """测试解释命令处理器"""

    def test_command_attribute(self):
        """测试命令属性"""
        handler = ExplanationCommandHandler()
        assert handler.command == "explanation"

    def test_get_template(self):
        """测试获取模板"""
        handler = ExplanationCommandHandler()
        template = handler.get_template()

        assert "{{ content }}" in template
        assert "解释" in template

    @patch("aidev_agent.services.command_handler.SafeJinjaEnvironment")
    def test_process_content(self, mock_jinja_env):
        """测试内容处理"""
        mock_env_instance = Mock()
        mock_env_instance.render.return_value = (
            "请解释以下内容Python是什么\n解释要求: 确保解释准确无误，无需冗余回答内容"
        )
        mock_jinja_env.return_value = mock_env_instance

        handler = ExplanationCommandHandler()
        context = [{"__key": "content", "__value": "Python是什么"}]

        result = handler.process_content(context)

        assert "解释" in result
        assert "Python" in result


class TestCommonCommandHandler:
    """测试通用命令处理器"""

    def test_init_with_parameters(self):
        """测试使用参数初始化"""
        handler = CommonCommandHandler("custom_command", "Custom template: {{ param }}")

        assert handler.command == "custom_command"
        assert handler.get_template() == "Custom template: {{ param }}"

    @pytest.mark.parametrize(
        "command,template",
        [
            ("summarize", "Summarize: {{ text }}"),
            ("analyze", "Analyze: {{ data }}"),
            ("format", "Format as {{ format_type }}: {{ content }}"),
        ],
    )
    def test_different_commands_and_templates(self, command, template):
        """测试不同的命令和模板"""
        handler = CommonCommandHandler(command, template)

        assert handler.command == command
        assert handler.get_template() == template


class TestCommonCommandHandlerBuilder:
    """测试通用命令处理器构建器"""

    def test_build_handler(self):
        """测试构建处理器"""
        handler = CommonCommandHandlerBuilder.build("test_command", "Test template: {{ value }}")

        assert isinstance(handler, CommonCommandHandler)
        assert handler.command == "test_command"
        assert handler.get_template() == "Test template: {{ value }}"

    @pytest.mark.parametrize(
        "command_id,template",
        [
            ("cmd1", "Template 1: {{ var1 }}"),
            ("cmd2", "Template 2: {{ var2 }} and {{ var3 }}"),
            ("cmd3", "Simple template"),
        ],
    )
    def test_build_different_handlers(self, command_id, template):
        """测试构建不同的处理器"""
        handler = CommonCommandHandlerBuilder.build(command_id, template)

        assert handler.command == command_id
        assert handler.get_template() == template


class TestCommandProcessor:
    """测试命令处理器"""

    def setup_method(self):
        """每个测试方法前的设置"""
        # 清理处理器注册表
        CommandProcessor._handlers.clear()

    def test_register_handler(self):
        """测试注册处理器"""
        CommandProcessor.register_handler("test", TranslateCommandHandler)

        assert "test" in CommandProcessor._handlers
        assert CommandProcessor._handlers["test"] == TranslateCommandHandler

    def test_register_multiple_handlers(self):
        """测试注册多个处理器"""
        CommandProcessor.register_handler("translate", TranslateCommandHandler)
        CommandProcessor.register_handler("explain", ExplanationCommandHandler)

        assert len(CommandProcessor._handlers) == 2
        assert CommandProcessor._handlers["translate"] == TranslateCommandHandler
        assert CommandProcessor._handlers["explain"] == ExplanationCommandHandler

    def test_process_command_success(self):
        """测试成功处理命令"""
        CommandProcessor.register_handler("translate", TranslateCommandHandler)

        processor = CommandProcessor()
        command_data = {
            "command": "translate",
            "context": [{"__key": "content", "__value": "Hello"}, {"__key": "language", "__value": "中文"}],
        }

        with patch.object(TranslateCommandHandler, "process_content", return_value="Processed content"):
            result = processor.process_command(command_data)

            assert result == "Processed content"

    def test_process_command_no_command(self):
        """测试处理没有命令的数据"""
        processor = CommandProcessor()
        command_data = {"context": []}

        with pytest.raises(ValueError, match="No command found in data"):
            processor.process_command(command_data)

    def test_process_command_empty_command(self):
        """测试处理空命令"""
        processor = CommandProcessor()
        command_data = {"command": "", "context": []}

        with pytest.raises(ValueError, match="No command found in data"):
            processor.process_command(command_data)

    def test_process_command_unregistered_handler(self):
        """测试处理未注册的命令"""
        processor = CommandProcessor()
        command_data = {"command": "unknown_command", "context": []}

        with pytest.raises(ValueError, match="No handler registered for command: unknown_command"):
            processor.process_command(command_data)

    def test_process_command_handler_error(self):
        """测试处理器执行错误"""
        CommandProcessor.register_handler("translate", TranslateCommandHandler)

        processor = CommandProcessor()
        command_data = {
            "command": "translate",
            "context": [{"__key": "language", "__value": "中文"}],  # 缺少content
        }

        with pytest.raises(ValueError, match="Translation requires 'content'"):
            processor.process_command(command_data)

    def test_process_command_missing_context(self):
        """测试处理缺少上下文的命令"""
        CommandProcessor.register_handler("translate", TranslateCommandHandler)

        processor = CommandProcessor()
        command_data = {"command": "translate"}  # 没有context

        with patch.object(TranslateCommandHandler, "process_content", return_value="Default result"):
            result = processor.process_command(command_data)

            # 验证传递了空列表作为上下文
            TranslateCommandHandler.process_content.assert_called_once_with([])
            assert result == "Default result"

    @pytest.mark.parametrize(
        "command_name,handler_class", [("translate", TranslateCommandHandler), ("explain", ExplanationCommandHandler)]
    )
    def test_process_different_commands(self, command_name, handler_class):
        """测试处理不同类型的命令"""
        CommandProcessor.register_handler(command_name, handler_class)

        processor = CommandProcessor()
        command_data = {"command": command_name, "context": [{"__key": "content", "__value": "test content"}]}

        with patch.object(handler_class, "process_content", return_value=f"Processed by {command_name}"):
            result = processor.process_command(command_data)

            assert result == f"Processed by {command_name}"
