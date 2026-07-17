from __future__ import annotations

import os
import sys
import tempfile
import types
from unittest.mock import MagicMock

import pytest

# Setup OpenAI mock classes for integration test
class MockChoiceMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

class MockChoice:
    def __init__(self, message):
        self.finish_reason = "stop"
        self.index = 0
        self.message = message

class MockChatCompletion:
    def __init__(self, id, model, choices):
        self.id = id
        self.model = model
        self.choices = choices
        self.usage = None

class MockCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, *args, **kwargs):
        self.last_kwargs = kwargs
        msg = MockChoiceMessage("assistant", "Mock response")
        return MockChatCompletion("msg-123", kwargs.get("model", "gpt-4"), [MockChoice(msg)])

# Dynamic registration of mock openai modules if not already done
if "openai" not in sys.modules:
    openai_mod = types.ModuleType("openai")
    sys.modules["openai"] = openai_mod
    
    openai_resources_mod = types.ModuleType("openai.resources")
    sys.modules["openai.resources"] = openai_resources_mod
    
    openai_chat_mod = types.ModuleType("openai.resources.chat")
    sys.modules["openai.resources.chat"] = openai_chat_mod
    
    openai_completions_mod = types.ModuleType("openai.resources.chat.completions")
    openai_completions_mod.Completions = MockCompletions
    sys.modules["openai.resources.chat.completions"] = openai_completions_mod

from sequa.cassette import cassette
from sequa.recorder import RecorderEngine
from sequa import storage


class MockLangChainMessage:
    def __init__(self, content: str):
        self.content = content

    def copy(self, update: dict | None = None) -> MockLangChainMessage:
        if update:
            return MockLangChainMessage(update.get("content", self.content))
        return MockLangChainMessage(self.content)


def test_pii_and_si_masking_unit():
    engine = RecorderEngine(path="dummy_path")

    # Test Emails
    assert engine.mask_pii_and_si("My email is test@example.com.") == "My email is [EMAIL]."
    assert engine.mask_pii_and_si("Contact info@company.co.uk for details.") == "Contact [EMAIL] for details."

    # Test Phone Numbers
    assert engine.mask_pii_and_si("Call me at 123-456-7890!") == "Call me at [PHONE]!"
    assert engine.mask_pii_and_si("Or call +1 (555) 555-5555.") == "Or call [PHONE]."
    assert engine.mask_pii_and_si("Number: +91 99999 88888.") == "Number: [PHONE]."

    # Test Credit Cards
    assert engine.mask_pii_and_si("My visa card is 1234-5678-9012-3456.") == "My visa card is [CREDIT_CARD]."
    assert engine.mask_pii_and_si("Amex: 1234 567890 12345.") == "Amex: [CREDIT_CARD]."

    # Test SSNs
    assert engine.mask_pii_and_si("SSN: ***-**-****.") == "SSN: ***-**-****." # Only matching standard format
    assert engine.mask_pii_and_si("My SSN is 123-45-6789.") == "My SSN is [SSN]."

    # Test IP Addresses
    assert engine.mask_pii_and_si("IP is 192.168.1.1.") == "IP is [IP_ADDRESS]."

    # Test API Keys
    assert engine.mask_pii_and_si("sk-1234567890abcdef12345678") == "[API_KEY]"
    assert engine.mask_pii_and_si("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q") == "[API_KEY]"

    # Test Bearer Tokens
    assert engine.mask_pii_and_si("Authorization: Bearer my-secret-token-123") == "Authorization: Bearer [TOKEN]"


def test_recursive_masking_unit():
    engine = RecorderEngine(path="dummy_path", mask_pii=True)

    # Test nested dict
    data = {
        "text": "Send email to test@example.com",
        "nested": {
            "phone": "555-555-5555",
            "number": 12345
        },
        "list": ["My IP is 10.0.0.1", 42]
    }
    masked = engine.mask_value(data)
    assert masked["text"] == "Send email to [EMAIL]"
    assert masked["nested"]["phone"] == "[PHONE]"
    assert masked["nested"]["number"] == 12345
    assert masked["list"][0] == "My IP is [IP_ADDRESS]"
    assert masked["list"][1] == 42

    # Test LangChain Message Mock
    msg = MockLangChainMessage("Hi, my email is admin@site.com")
    masked_msg = engine.mask_value(msg)
    assert isinstance(masked_msg, MockLangChainMessage)
    assert masked_msg.content == "Hi, my email is [EMAIL]"


def test_masking_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        from openai.resources.chat.completions import Completions
        
        last_kwargs = []
        original_create = Completions.create
        
        def spy_create(self, *args, **kwargs):
            last_kwargs.append(kwargs)
            return original_create(self, *args, **kwargs)
            
        Completions.create = spy_create
        
        try:
            completions_mock = Completions()
            
            # Verify without masking first (mask_pii=False by default)
            with cassette(tmpdir, mode="record", adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()) as cas:
                res = completions_mock.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Contact me at alice@example.com."}],
                )
                
                # The mock completions object should have received the raw email
                assert last_kwargs[-1]["messages"][0]["content"] == "Contact me at alice@example.com."

            # Clear cassettes for the next run
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))

            # Now test with mask_pii=True
            with cassette(tmpdir, mode="record", mask_pii=True, adapter=sys.modules["sequa.llm.adapters"].OpenAIAdapter()) as cas:
                res = completions_mock.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Contact me at bob@example.com or call 555-555-5555."}],
                )
                
                # The mock completions object should have received the masked email/phone
                assert last_kwargs[-1]["messages"][0]["content"] == "Contact me at [EMAIL] or call [PHONE]."

            # Verify that the recorded cassette file also contains the masked content
            files = os.listdir(tmpdir)
            assert len(files) == 1
            cassette_path = os.path.join(tmpdir, files[0])
            cassette_obj = storage.load(cassette_path)
            
            # Verify the recorded request matches the masked content
            assert cassette_obj.request["messages"][0]["content"] == "Contact me at [EMAIL] or call [PHONE]."
        finally:
            Completions.create = original_create
