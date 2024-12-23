from typing import Dict, Any, Optional
from pathlib import Path
from ..llm_recorder import LLMRecorder

from pprint import pprint

try:
    import anthropic
    from anthropic import Anthropic
    from anthropic.types import Message
except ImportError:
    raise ImportError(
        "Anthropic provider requires the 'anthropic' package. "
        "Install it with 'pip install llm-recorder[anthropic]'"
    )


class AnthropicReplayLLM(LLMRecorder):
    """Implementation of LLMRecorder for Anthropic"""

    def __init__(self, original_create, **kwargs):
        super().__init__(**kwargs)
        self._original_create = original_create

    def make_live_call(self, **kwargs) -> Message:
        """Make a live API call to Anthropic"""
        pprint(kwargs)
        response = self._original_create(**kwargs)
        return response

    def dict_to_model_response(self, dict_response: Dict[str, Any]) -> Message:
        """Convert a dictionary back to a model response object"""
        return Message.model_validate(dict_response)

    def model_response_to_dict(self, model_response: Message) -> Dict[str, Any]:
        """Convert a model response object to a dictionary"""
        return model_response.model_dump()


class ReplayMessages:
    """Wrapper for Anthropic messages that uses LLMRecorder for create calls"""

    def __init__(
        self,
        client: Anthropic,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
    ):
        self._client = client
        # Store the original create method
        self._original_create = client.messages.create
        self._replay_llm = AnthropicReplayLLM(
            original_create=self._original_create,
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count,
        )

    def create(self, **kwargs) -> Message:
        """Create a message with replay support"""
        response = self._replay_llm.completion(**kwargs)
        return response


class ReplayAnthropic(Anthropic):
    """Anthropic client that supports replaying messages"""

    def __init__(
        self,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
        **kwargs,
    ):
        """
        Initialize ReplayAnthropic.

        Args:
            replay_dir: Directory to load interactions from
            save_dir: Optional directory to save new interactions. If None, saves to replay_dir
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to Anthropic client
        """
        super().__init__(**kwargs)

        # Create messages instance with replay support
        self.messages = ReplayMessages(
            client=self,
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count,
        )
