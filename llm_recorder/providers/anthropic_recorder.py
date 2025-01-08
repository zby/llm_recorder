from typing import Dict, Any, Union
from pathlib import Path
from ..llm_recorder import LLMRecorder, Persistence

try:
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
        persistence: Union[str, Path, Persistence],
        replay_count: int = 0,
    ):
        super().__init__(client=client)
        LLMRecorder.__init__(
            self,
            persistence,
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
        persistence: Union[str, Path, Persistence],
        replay_count: int = 0,
        **kwargs,
    ):
        """
        Initialize ReplayAnthropic.

        Args:
            persistence: Either a Persistence implementation or a path (str/Path) for default FilePersistence
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to Anthropic client
        """
        super().__init__(**kwargs)

        # Create messages instance with replay support
        self.messages = ReplayMessages(
            client=self,
            persistence=persistence,
            replay_count=replay_count,
        )
