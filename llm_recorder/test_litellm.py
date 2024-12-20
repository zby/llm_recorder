import pytest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, MagicMock
from .litellm_replay import ReplayLiteLLM, LLMInteraction

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

@pytest.fixture
def sample_request():
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}]
    }

@pytest.fixture
def sample_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

def create_interaction_files(directory: Path, request: dict, response: dict):
    """Helper to create interaction files in a directory"""
    with open(directory / "1.request.json", "w") as f:
        json.dump(request, f)
    with open(directory / "1.response.json", "w") as f:
        json.dump(response, f)

def test_replay_llm_initialization(temp_dir):
    replay_llm = ReplayLiteLLM(replay_dir=temp_dir)
    assert replay_llm.replay_dir == temp_dir
    assert replay_llm.save_dir == temp_dir

def test_replay_llm_with_invalid_replay_dir():
    with pytest.raises(FileNotFoundError):
        ReplayLiteLLM(replay_dir="nonexistent", replay_count=1)

def test_replay_llm_loads_interactions(temp_dir, sample_request, sample_response):
    create_interaction_files(temp_dir, sample_request, sample_response)
    
    replay_llm = ReplayLiteLLM(replay_dir=temp_dir, replay_count=1)
    assert len(replay_llm.interactions) == 1
    assert replay_llm.interactions[0].request == sample_request
    assert replay_llm.interactions[0].response == sample_response

def test_replay_llm_makes_live_call_when_no_replay(temp_dir, sample_request, sample_response):
    mock_response = MagicMock()
    mock_response.model_dump.return_value = sample_response
    mock_completion = MagicMock()
    mock_completion.return_value = mock_response
    
    replay_llm = ReplayLiteLLM(replay_dir=temp_dir, completion_function=mock_completion)
    response = replay_llm.completion(**sample_request)
    
    mock_completion.assert_called_once_with(**sample_request)
    assert response.choices[0].message.content == "Hello there!"

def test_replay_llm_uses_replay_when_available(temp_dir, sample_request, sample_response):
    create_interaction_files(temp_dir, sample_request, sample_response)
    
    replay_llm = ReplayLiteLLM(replay_dir=temp_dir, replay_count=1)
    response = replay_llm.completion(**sample_request)
    
    assert response.choices[0].message.content == "Hello there!"

def test_save_and_load_interaction(temp_dir):
    interaction = LLMInteraction(
        timestamp="2024-01-01T00:00:00",
        request={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello"}]},
        response={"choices": [{"message": {"content": "Hi!"}}]}
    )
    
    interaction.save_to_directory(temp_dir, 1)
    loaded = LLMInteraction.load_from_directory(temp_dir, 1)
    
    assert loaded.request == interaction.request
    assert loaded.response == interaction.response