from typing import Dict, Any, Optional
from pathlib import Path
from ..llm_recorder import LLMRecorder

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


class OpenAILLMRecorder(LLMRecorder):
    """Implementation of LLMRecorder for OpenAI"""
    def __init__(self, original_create, **kwargs):
        super().__init__(**kwargs)
        self._original_create = original_create

    def make_live_call(self, **kwargs) -> ChatCompletion:
        """Make a live API call to OpenAI"""
        response = self._original_create(**kwargs)
        return response
    
    def dict_to_model_response(self, dict_response: Dict[str, Any]) -> ChatCompletion:
        """Convert a dictionary back to a model response object"""
        return ChatCompletion.model_validate(dict_response)

    def model_response_to_dict(self, model_response: ChatCompletion) -> Dict[str, Any]:
        """Convert a model response object to a dictionary"""
        return model_response.model_dump()


class CompletionsRecorder(completions.Completions):
    """Subclass of OpenAI Completions that uses LLMRecorder for create calls"""
    def __init__(self, client: OpenAI, replay_dir: str | Path, save_dir: Optional[str | Path] = None, replay_count: int = 0, **kwargs):
        super().__init__(client=client, **kwargs)
        # Store the original create method before we override it
        original_create = super().create
        self._replay_llm = OpenAILLMRecorder(
            original_create=original_create,
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count
        )

    def create(self, **kwargs) -> ChatCompletion:
        response = self._replay_llm.completion(**kwargs)
        return response


class ChatRecorder(chat.Chat):
    """Subclass of OpenAI Chat that uses LLMRecorder for create calls"""
    def __init__(self,
                 client: OpenAI,
                 replay_dir: str | Path,
                 save_dir: Optional[str | Path] = None,
                 replay_count: int = 0):
        super().__init__(client=client)
        self._replay_dir = replay_dir
        self._save_dir = save_dir
        self._replay_count = replay_count

    @cached_property
    def completions(self) -> CompletionsRecorder:
        return CompletionsRecorder(
            self._client,
            replay_dir=self._replay_dir,
            save_dir=self._save_dir,
            replay_count=self._replay_count
        )


class OpenAIRecorder(OpenAI):
    """OpenAI client that supports replaying chat completions"""
    def __init__(
        self,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
        **kwargs
    ):
        """
        Initialize ReplayOpenAI.
        
        Args:
            replay_dir: Directory to load interactions from
            save_dir: Optional directory to save new interactions. If None, saves to replay_dir
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to OpenAI client
        """
        super().__init__(**kwargs)
        
        # Create new chat instance with replay support
        self.chat = ChatRecorder(
            client=self,
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count,
            **kwargs
        )
