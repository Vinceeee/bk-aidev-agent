from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


class ExecuteKwargs(BaseModel):
    stream: bool = False
    stream_timeout: int = 30
    passthrough_input: bool = False
    run_agent: bool = False


class SessionTool(BaseModel):
    tool_id: int
    tool_code: str
    icon: str
    tool_name: str = Field(validation_alias=AliasChoices("tool_name", "tool_cn_name"))
    description: str
    is_sensitive: bool
    status: Literal["ready", "deleted"] = "ready"
    property: dict = Field(default_factory=dict)

    @classmethod
    def get_model_fields_list_without_default_values(cls) -> list[str]:
        field_list = []
        for name, field_info in cls.model_fields.items():
            if field_info.default:
                continue
            field_list.append(name)
        return field_list


class SessionContentExtra(BaseModel):
    """会话内容的一些额外属性"""

    tools: list[SessionTool] = Field(default_factory=list)
    anchor_path_resources: dict = Field(default_factory=dict)
    context: list[dict] | None = None
    command: str | None = None
    rendered_content: str | None = None


class SessionContentProperty(BaseModel):
    """会话内容的一些额外属性"""

    extra: SessionContentExtra | None = None


class ChatPrompt(BaseModel):
    id: int | None = None
    role: str
    content: str
    extra: SessionContentExtra | None = None

    @model_validator(mode="before")
    def validate_content_with_rendered(cls, values: Any) -> Any:
        extra = values.get("extra")
        if extra:
            if isinstance(extra, dict):
                rendered_content = extra.get("rendered_content")
                if rendered_content:
                    values["content"] = rendered_content
            elif hasattr(extra, "rendered_content") and extra.rendered_content:
                values["content"] = extra.rendered_content
        return values
