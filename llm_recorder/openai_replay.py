from typing import Optional, Dict, Any
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletion
from .litellm_replay import ReplayLiteLLM

class OpenAIReplayLLM(ReplayLiteLLM):
    def __init__(
        self,
        client: OpenAI,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
    ):
        """
        Initialize OpenAIReplayLLM.
        
        Args:
            client: Original OpenAI client
            replay_dir: Directory to load interactions from
            save_dir: Optional directory to save new interactions. If None, saves to replay_dir
            replay_count: Number of interactions to replay.
                        After these are exhausted, live API calls will be made.
        """
        self.client = client
        super().__init__(
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count,
            completion_function=self.client.chat.completions.create
        )

    def dict_to_model_response(self, dict_response: Dict[str, Any]) -> ChatCompletion:
        """Convert a dictionary back to an OpenAI response object"""
        return ChatCompletion(**dict_response)
    
    def model_response_to_dict(self, model_response: ChatCompletion) -> Dict[str, Any]:
        """Convert an OpenAI response object to a dictionary"""
        return model_response.model_dump()

    def completion(self, **kwargs) -> ChatCompletion:
        """Override to ensure proper typing"""
        return super().completion(**kwargs)


def openai_enable_replay_mode(
    client: OpenAI,
    replay_dir: str | Path,
    save_dir: Optional[str | Path] = None,
    replay_count: int = 0,
) -> OpenAI:
    """
    Enable replay mode for an OpenAI client.
    
    Args:
        client: OpenAI client to wrap
        replay_dir: Directory to load interactions from
        save_dir: Directory to save new interactions. Defaults to replay_dir if None
        replay_count: Number of interactions to replay before making live calls
        
    Returns:
        The OpenAI client with chat.completions.create wrapped for replay
    """
    replay_llm = OpenAIReplayLLM(
        client=client,
        replay_dir=replay_dir,
        save_dir=save_dir,
        replay_count=replay_count,
    )
    
    # Replace the chat completions create method with our replay version
    client.chat.completions.create = replay_llm.completion
    
    return client