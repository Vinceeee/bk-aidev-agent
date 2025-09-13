# AGENTS.md

## 项目概述
- 这是一个用于构建 bk-aidev 智能体的 Python 工具包

## 使用指南
- 使用 `uv run pytest` 而不是 `pytest` 来运行测试
- 完成了任务后,请执行`git commit`保存当前变更
- 提交信息应该类似于 `feat: add a new feature` 或 `fix: fix a bug`，必须以 "committed by AI assistant" 结尾

## 项目结构
- `tests/` 包含所有测试用例
- `aidev_agent/` 包含主要源代码
- `aidev_agent/api` 为 bk-aidev web api 提供 HTTP 函数接口
- `aidev_agent/packages` 被所有其他模块使用，主要供 SDK 用户使用，目前扩展了一些 langchain 模块，如 `tool` 和 `chat_models`
- `aidev_agent/schemas` 包含 SDK 的所有模式定义
- `aidev_agent/utils` 包含一些难以分类的实用工具函数

## 单元测试编写规则
- 尽量使用参数化单元测试,减少重复代码
- 单元测试函数不能超过30行