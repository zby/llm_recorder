from typing import Dict, Any, Optional
from pathlib import Path
from ..llm_recorder import LLMRecorder

from pprint import pprint

try:
    import anthropic
    from anthropic import Anthropic
    from anthropic.resources.messages import Messages
    from anthropic.types import Message
except ImportError:
    raise ImportError(
        "Anthropic provider requires the 'anthropic' package. "
        "Install it with 'pip install llm-recorder[anthropic]'"
    )


class ReplayMessages(Messages, LLMRecorder):
    """Wrapper for Anthropic messages that uses LLMRecorder to replay messages"""

    def __init__(
        self,
        client: Anthropic,
        store_path: str | Path,
        replay_count: int = 0,
    ):
        super().__init__(client=client)
        LLMRecorder.__init__(
            self,
            store_path=store_path,
            replay_count=replay_count,
        )

    def live_call(self, **kwargs) -> Message:
        """Make a live API call to Anthropic"""
        response = super().create(**kwargs)
        return response

    def req_to_dict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return req

    def res_to_dict(self, res: Message) -> Dict[str, Any]:
        return res.model_dump()

    def create(self, **kwargs) -> Message:
        """Create a message with replay support"""
        dict_response = self.dict_completion(**kwargs)
        return Message.model_validate(dict_response)


class ReplayAnthropic(Anthropic):
    """Anthropic client that supports replaying messages"""

    def __init__(
        self,
        store_path: str | Path,
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
            store_path=store_path,
            replay_count=replay_count,
        )
