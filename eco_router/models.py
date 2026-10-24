from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Tier = Literal['eco', 'balanced', 'heavy']


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['system', 'developer', 'user', 'assistant', 'tool']
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'eco-router'
    messages: list[Message] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: Literal[1] = 1
    user: str | None = None
    cache: bool = True

    @model_validator(mode='after')
    def valid_request(self):
        if self.model != 'eco-router':
            raise ValueError('Use the virtual model eco-router; configure upstream models on the server.')
        if sum(len(m.content or '') for m in self.messages) > 100_000:
            raise ValueError('Conversation exceeds 100,000 characters.')
        if not any(m.content or m.tool_calls for m in self.messages):
            raise ValueError('Conversation cannot be empty.')
        if self.response_format and self.response_format.get('type') not in {'text', 'json_object', 'json_schema'}:
            raise ValueError('Unsupported response_format type.')
        if self.response_format and self.response_format.get('type') == 'json_schema':
            from jsonschema import Draft202012Validator, SchemaError
            try:
                schema = self.response_format['json_schema']['schema']
                Draft202012Validator.check_schema(schema)
            except (KeyError, TypeError, SchemaError) as exc:
                raise ValueError('response_format requires a valid JSON schema.') from exc
        return self


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid')
    task_id: int = Field(ge=1)
    prompt_segment: str = Field(min_length=1, max_length=100_000)
    target_tier: Tier
    dependencies: list[int] = Field(default_factory=list)
