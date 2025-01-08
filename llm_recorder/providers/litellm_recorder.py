from typing import Optional, Dict, Any, Union
import litellm
from llm_recorder import LLMRecorder, Persistence
from pathlib import Path

# this is for monkey patching
_rllm_instance: Optional["LitellmRecorder"] = None

# Store the original completion function
_original_completion = litellm.completion


class LitellmRecorder(LLMRecorder):
    """
    A concrete implementation of LLMRecorder for LiteLLM.
    """

    # first we need to implement the live_call, req_to_dict and res_to_dict methods
    # that are abstract in the LLMRecorder class

    def live_call(self, **kwargs) -> litellm.ModelResponse:
        """Make a live call to the LLM using the original completion function."""
        return _original_completion(**kwargs)

    def req_to_dict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Convert request to a dictionary format."""
        return req

    def res_to_dict(self, res: litellm.ModelResponse) -> Dict[str, Any]:
        """Convert LiteLLM response to a dictionary format."""
        return res.model_dump()

    # then we can implement the completion method for convenience

    def completion(self, **kwargs) -> litellm.ModelResponse:
        dict_response = self.dict_completion(**kwargs)
        litellm_message = litellm.ModelResponse(**dict_response)
        return litellm_message


def enable_replay_mode(
    persistence: Union[str, Path, Persistence],
    replay_count: int = 0,
) -> None:
    """
    Enable replay mode by creating a LiteLLMRecorder instance and monkey-patching litellm.completion.

    Args:
        persistence: Directory to load interactions from or a Persistence implementation.
        replay_count: Number of interactions to replay before making live calls.
    """
    global _rllm_instance

    if _rllm_instance is not None:
        # Already enabled, do nothing
        return

    _rllm_instance = LitellmRecorder(
        persistence=persistence,
        replay_count=replay_count,
    )

    def patched_completion(**kwargs):
        return _rllm_instance.completion(**kwargs)

    litellm.completion = patched_completion
