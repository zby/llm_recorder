import pytest
from pathlib import Path
import json
import tempfile
from llm_recorder.llm_recorder import LLMRecorder, LLMInteraction, FilePersistence
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_request():
    return {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }


@pytest.fixture
def sample_response():
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def create_interaction_files(directory: Path, request: dict, response: dict):
    """Helper to create interaction files in a directory"""
    persistence = FilePersistence(directory)
    interaction = LLMInteraction(
        timestamp=datetime.now().isoformat(),
        request=request,
        response=response
    )
    persistence.save(interaction)


def test_save_and_load_interaction(temp_dir):
    persistence = FilePersistence(temp_dir)
    
    interaction = LLMInteraction(
        timestamp="2024-01-01T00:00:00",
        request={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        response={"choices": [{"message": {"content": "Hi!"}}]},
    )

    # Save the interaction
    persistence.save(interaction)

    # Load all interactions (limit=1 since we only saved one)
    loaded_interactions = persistence.load_all(limit=1)
    
    assert len(loaded_interactions) == 1
    loaded = loaded_interactions[0]

    # Compare the dictionaries
    assert loaded.request == interaction.request
    assert loaded.response == interaction.response


@dataclass
class MockResponse:
    """Mock response object to simulate an LLM response"""

    content: str
    model: str = "mock-model"


class MockReplayLLM(LLMRecorder):
    """Test implementation of LLMRecorder"""

    def live_call(self, **kwargs) -> dict:
        """Simulate a live API call"""
        return {"choices": [{"message": {"content": "This is a live response"}}]}

    def req_to_dict(self, req: Any) -> Dict[str, Any]:
        """Convert request to dictionary"""
        return req

    def res_to_dict(self, res: Any) -> Dict[str, Any]:
        """Convert response to dictionary"""
        return res


def test_replay_llm_replay_mode(temp_dir, sample_request, sample_response):
    # Create interaction files
    create_interaction_files(temp_dir, sample_request, sample_response)

    # Initialize LLMRecorder in replay mode
    llm = MockReplayLLM(store_path=temp_dir, replay_count=1)

    # Get response - should be from replay
    response = llm.dict_completion(**sample_request)

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "Hello there!"


def test_replay_llm_live_mode(temp_dir):
    # Initialize LLMRecorder with no replay interactions
    llm = MockReplayLLM(store_path=temp_dir, replay_count=0)

    # Get response - should be live
    response = llm.dict_completion(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "This is a live response"


def test_replay_llm_saves_interactions(temp_dir, sample_request):
    # Initialize LLMRecorder
    llm = MockReplayLLM(store_path=temp_dir, replay_count=0)

    # Make a call
    llm.dict_completion(**sample_request)

    # Check that files were saved
    assert list(temp_dir.glob("1.request_*.json"))  # Should find at least one request file
    assert list(temp_dir.glob("1.response_*.json"))  # Should find at least one response file


def test_replay_llm_invalid_replay_count(temp_dir, sample_request, sample_response):
    # Create one interaction
    create_interaction_files(temp_dir, sample_request, sample_response)

    # Try to replay more interactions than exist
    with pytest.raises(ValueError, match="replay_count .* > available interactions"):
        MockReplayLLM(store_path=temp_dir, replay_count=2)
