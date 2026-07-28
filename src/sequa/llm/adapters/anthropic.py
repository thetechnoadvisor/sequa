from __future__ import annotations

from typing import Any
from sequa.llm.adapters.base import CanonicalRequest, CanonicalResponse, ProviderAdapter
from sequa.utils import serialize_type_or_pydantic


class AnthropicAdapter(ProviderAdapter):
    provider_name: str = "anthropic"

    def to_canonical_request(self, request: Any, **kwargs: Any) -> CanonicalRequest:
        model = kwargs.get("model")
        messages = kwargs.get("messages") or []
        temperature = kwargs.get("temperature")
        
        params = {k: v for k, v in kwargs.items() if k not in ("model", "messages", "temperature")}
        
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                content_val = serialize_type_or_pydantic(msg.get("content"))
                normalized_messages.append({"role": msg.get("role"), "content": content_val})
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "content", "")
                normalized_messages.append({"role": role, "content": serialize_type_or_pydantic(content)})

        return CanonicalRequest(
            provider=self.provider_name,
            model=model,
            temperature=temperature,
            messages=normalized_messages,
            params=serialize_type_or_pydantic(params),
            metadata={"raw_request": kwargs},
        )

    def to_canonical_response(self, response: Any, **kwargs: Any) -> CanonicalResponse:
        # Check if response is a dict or Anthropic Message object
        if isinstance(response, dict):
            resp_id = response.get("id")
            model = response.get("model") or kwargs.get("model")
            role = response.get("role", "assistant")
            stop_reason = response.get("stop_reason")
            stop_sequence = response.get("stop_sequence")
            raw_content = response.get("content") or []

            # Extract usage
            raw_usage = response.get("usage") or {}
            usage = {
                "input_tokens": raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens") or 0,
                "output_tokens": raw_usage.get("output_tokens") or raw_usage.get("completion_tokens") or 0,
            }
        else:
            # It's an Anthropic Message object
            resp_id = getattr(response, "id", None)
            model = getattr(response, "model", None) or kwargs.get("model")
            role = getattr(response, "role", "assistant")
            stop_reason = getattr(response, "stop_reason", None)
            stop_sequence = getattr(response, "stop_sequence", None)
            raw_content = getattr(response, "content", [])

            usage_obj = getattr(response, "usage", None)
            usage = None
            if usage_obj:
                usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", 0),
                    "output_tokens": getattr(usage_obj, "output_tokens", 0),
                }

        content_text = ""
        tool_calls = []
        content_serializable = []

        if isinstance(raw_content, list):
            for item in raw_content:
                if isinstance(item, dict):
                    b_type = item.get("type", "text")
                    if b_type == "text":
                        text_val = item.get("text", "")
                        content_text += text_val
                        content_serializable.append({
                            "type": "text",
                            "text": text_val,
                        })
                    elif b_type == "tool_use":
                        tc_info = {
                            "type": "tool_use",
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "input": item.get("input", {}),
                            "args": item.get("input", {}),
                        }
                        tool_calls.append(tc_info)
                        content_serializable.append({
                            "type": "tool_use",
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "input": item.get("input", {}),
                        })
                    else:
                        content_serializable.append(serialize_type_or_pydantic(item))
                else:
                    b_type = getattr(item, "type", "text")
                    if b_type == "text":
                        text_val = getattr(item, "text", "")
                        content_text += text_val
                        content_serializable.append({
                            "type": "text",
                            "text": text_val,
                        })
                    elif b_type == "tool_use":
                        item_id = getattr(item, "id", None)
                        item_name = getattr(item, "name", None)
                        item_input = getattr(item, "input", {})
                        tc_info = {
                            "type": "tool_use",
                            "id": item_id,
                            "name": item_name,
                            "input": item_input,
                            "args": item_input,
                        }
                        tool_calls.append(tc_info)
                        content_serializable.append({
                            "type": "tool_use",
                            "id": item_id,
                            "name": item_name,
                            "input": item_input,
                        })
                    else:
                        content_serializable.append(serialize_type_or_pydantic(item))
        else:
            content_text = str(raw_content or "")
            content_serializable.append({
                "type": "text",
                "text": content_text,
            })

        if tool_calls and not stop_reason:
            stop_reason = "tool_use"

        return CanonicalResponse(
            provider=self.provider_name,
            model=model,
            output=content_text,
            usage=usage,
            tool_calls=tool_calls,
            metadata={
                "raw_response": {
                    "id": resp_id,
                    "model": model,
                    "role": role,
                    "type": "message",
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                    "content": content_serializable,
                }
            }
        )

    def from_canonical_response(self, response: CanonicalResponse, request: Any, is_parse: bool = False, **kwargs: Any) -> Any:
        raw_resp = response.metadata.get("raw_response", {})
        resp_id = raw_resp.get("id") or "replayed-response"
        content_data = raw_resp.get("content") or []
        usage_data = response.usage or {}

        replayed_msg = None
        try:
            from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
            
            content_blocks = []
            for item in content_data:
                if isinstance(item, dict):
                    b_type = item.get("type", "text")
                    if b_type == "text":
                        content_blocks.append(TextBlock(
                            text=item.get("text", response.output or ""),
                            type="text"
                        ))
                    elif b_type == "tool_use":
                        content_blocks.append(ToolUseBlock(
                            id=item.get("id", "toolu_default"),
                            name=item.get("name", ""),
                            input=item.get("input", {}),
                            type="tool_use"
                        ))
                    else:
                        content_blocks.append(TextBlock(
                            text=str(item),
                            type="text"
                        ))
                else:
                    content_blocks.append(item)
                
            usage_obj = Usage(
                input_tokens=usage_data.get("input_tokens", 0) if usage_data else 0,
                output_tokens=usage_data.get("output_tokens", 0) if usage_data else 0,
            )

            stop_reason = raw_resp.get("stop_reason")
            if not stop_reason:
                stop_reason = "tool_use" if response.tool_calls else "end_turn"

            replayed_msg = Message(
                id=resp_id,
                content=content_blocks,
                model=response.model or "replayed-model",
                role=raw_resp.get("role", "assistant"),
                stop_reason=stop_reason,
                stop_sequence=raw_resp.get("stop_sequence"),
                type="message",
                usage=usage_obj,
            )
        except Exception:
            class MockTextBlock:
                def __init__(self, text: str):
                    self.text = text
                    self.type = "text"

            class MockToolUseBlock:
                def __init__(self, id: str, name: str, input: dict[str, Any]):
                    self.id = id
                    self.name = name
                    self.input = input
                    self.type = "tool_use"
                    
            class MockUsage:
                def __init__(self, input_t: int, output_t: int):
                    self.input_tokens = input_t
                    self.output_tokens = output_t
                    
            class MockMessage:
                def __init__(self, id: str, content: list[Any], model: str, role: str, stop_reason: str, stop_sequence: str | None, usage: MockUsage | None):
                    self.id = id
                    self.content = content
                    self.model = model
                    self.role = role
                    self.stop_reason = stop_reason
                    self.stop_sequence = stop_sequence
                    self.type = "message"
                    self.usage = usage
                    
            content_blocks = []
            for item in content_data:
                if isinstance(item, dict):
                    b_type = item.get("type", "text")
                    if b_type == "text":
                        content_blocks.append(MockTextBlock(text=item.get("text", response.output or "")))
                    elif b_type == "tool_use":
                        content_blocks.append(MockToolUseBlock(
                            id=item.get("id", "toolu_default"),
                            name=item.get("name", ""),
                            input=item.get("input", {}),
                        ))
                    else:
                        content_blocks.append(MockTextBlock(text=str(item)))
                else:
                    content_blocks.append(item)
                
            usage_obj = None
            if usage_data:
                usage_obj = MockUsage(
                    usage_data.get("input_tokens") or usage_data.get("prompt_tokens") or 0,
                    usage_data.get("output_tokens") or usage_data.get("completion_tokens") or 0,
                )
                
            stop_reason = raw_resp.get("stop_reason")
            if not stop_reason:
                stop_reason = "tool_use" if response.tool_calls else "end_turn"

            replayed_msg = MockMessage(
                id=resp_id,
                content=content_blocks,
                model=response.model or "replayed-model",
                role=raw_resp.get("role", "assistant"),
                stop_reason=stop_reason,
                stop_sequence=raw_resp.get("stop_sequence"),
                usage=usage_obj,
            )

        output_format = kwargs.get("output_format")
        if is_parse or output_format:
            try:
                import anthropic.resources.messages.messages as mod
                return mod.parse_response(response=replayed_msg, output_format=output_format)
            except Exception:
                pass

            if hasattr(replayed_msg, "content") and replayed_msg.content:
                text_block = replayed_msg.content[0]
                parsed_val = None
                text_val = getattr(text_block, "text", "")
                if text_val and hasattr(output_format, "model_validate_json"):
                    try:
                        parsed_val = output_format.model_validate_json(text_val)
                    except Exception:
                        pass
                if not hasattr(text_block, "parsed_output"):
                    setattr(text_block, "parsed_output", parsed_val)

        return replayed_msg

