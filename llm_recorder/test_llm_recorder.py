import pytest
from pathlib import Path
import json
import tempfile
from llm_recorder import LLMRecorder, LLMInteraction
from dataclasses import dataclass


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
    with open(directory / "1.request.json", "w") as f:
        json.dump(request, f)
    with open(directory / "1.response.json", "w") as f:
        json.dump(response, f)


def test_save_and_load_interaction(temp_dir):
    interaction = LLMInteraction(
        timestamp="2024-01-01T00:00:00",
        request={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        response={"choices": [{"message": {"content": "Hi!"}}]},
    )

    interaction.save_to_directory(temp_dir, 1)
    loaded = LLMInteraction.load_from_directory(temp_dir, 1)

    assert loaded.request == interaction.request
    assert loaded.response == interaction.response


@dataclass
class MockResponse:
    """Mock response object to simulate an LLM response"""

    content: str
    model: str = "mock-model"


class MockReplayLLM(LLMRecorder):
    """Test implementation of ReplayLLM"""

    def make_live_call(self, **kwargs) -> MockResponse:
        """Simulate a live API call"""
        return MockResponse(content="This is a live response")

    def dict_to_model_response(self, dict_response: dict) -> MockResponse:
        """Convert dictionary to MockResponse"""
        return MockResponse(
            content=dict_response["choices"][0]["message"]["content"],
            model=dict_response.get("model", "mock-model"),
        )

    def model_response_to_dict(self, model_response: MockResponse) -> dict:
        """Convert MockResponse to dictionary"""
        return {
            "model": model_response.model,
            "choices": [{"message": {"content": model_response.content}}],
        }


def test_replay_llm_replay_mode(temp_dir, sample_request, sample_response):
    # Create interaction files
    create_interaction_files(temp_dir, sample_request, sample_response)

    # Initialize ReplayLLM in replay mode
    llm = MockReplayLLM(replay_dir=temp_dir, replay_count=1)

    # Get response - should be from replay
    response = llm.completion(**sample_request)

    assert isinstance(response, MockResponse)
    assert response.content == "Hello there!"


def test_replay_llm_live_mode(temp_dir):
    # Initialize ReplayLLM with no replay interactions
    llm = MockReplayLLM(replay_dir=temp_dir, replay_count=0)

    # Get response - should be live
    response = llm.completion(messages=[{"role": "user", "content": "Hi"}])

    assert isinstance(response, MockResponse)
    assert response.content == "This is a live response"


def test_replay_llm_saves_interactions(temp_dir, sample_request):
    # Initialize ReplayLLM
    llm = MockReplayLLM(replay_dir=temp_dir, replay_count=0)

    # Make a call
    llm.completion(**sample_request)

    # Check that files were saved
    assert (temp_dir / "1.request.json").exists()
    assert (temp_dir / "1.response.json").exists()


def test_replay_llm_invalid_replay_count(temp_dir, sample_request, sample_response):
    # Create one interaction
    create_interaction_files(temp_dir, sample_request, sample_response)

    # Try to replay more interactions than exist
    with pytest.raises(ValueError, match="replay_count is greater than"):
        MockReplayLLM(replay_dir=temp_dir, replay_count=2)
