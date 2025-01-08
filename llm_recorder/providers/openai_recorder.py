from typing import Dict, Any, Union
from pathlib import Path
from ..llm_recorder import LLMRecorder, Persistence

from functools import cached_property

try:
    from openai import OpenAI
    from openai.resources import chat
    from openai.resources.chat import completions
    from openai.types.chat import ChatCompletion
except ImportError:
    raise ImportError(
        "OpenAI provider requires the 'openai' package. "
        "Install it with 'pip install llm-recorder[openai]'"
    )


class CompletionsRecorder(completions.Completions, LLMRecorder):
    """Subclass of OpenAI Completions that uses LLMRecorder for create calls"""

    def __init__(
        self,
        client: OpenAI,
        persistence: Union[str, Path, Persistence],
        replay_count: int = 0,
        **kwargs,
    ):
        super().__init__(client=client, **kwargs)
        LLMRecorder.__init__(
            self,
            persistence,
            replay_count=replay_count,
        )

    def live_call(self, **kwargs) -> ChatCompletion:
        """Make a live API call to OpenAI"""
        response = super().create(**kwargs)
        return response

    def create(self, **kwargs) -> ChatCompletion:
        dict_response = self.dict_completion(**kwargs)
        return ChatCompletion.model_validate(dict_response)

    def req_to_dict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        return req

    def res_to_dict(self, res: ChatCompletion) -> Dict[str, Any]:
        return res.model_dump()


class ChatRecorder(chat.Chat):
    """Subclass of OpenAI Chat that uses LLMRecorder for create calls"""

    def __init__(
        self,
        client: OpenAI,
        persistence: Union[str, Path, Persistence],
        replay_count: int = 0,
    ):
        super().__init__(client=client)
        self._persistence = persistence
        self._replay_count = replay_count

    @cached_property
    def completions(self) -> CompletionsRecorder:
        return CompletionsRecorder(
            self._client,
            self._persistence,
            replay_count=self._replay_count,
        )


class OpenAIRecorder(OpenAI):
    """OpenAI client that supports replaying chat completions"""

    def __init__(
        self,
        persistence: Union[str, Path, Persistence],
        replay_count: int = 0,
        **kwargs,
    ):
        """
        Initialize ReplayOpenAI.

        Args:
            persistence: Either a Persistence implementation or a path (str/Path) for default FilePersistence
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(**kwargs)

        # Create new chat instance with replay support
        self.chat = ChatRecorder(
            client=self,
            persistence=persistence,
            replay_count=replay_count,
        )
