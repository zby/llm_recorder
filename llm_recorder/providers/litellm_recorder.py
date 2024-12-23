from typing import Dict, Any, Optional
import litellm
from ..llm_recorder import LLMRecorder
from pathlib import Path

# this is for monkey patching
_rllm_instance: Optional["LLMRecorder"] = None


def enable_replay_mode(
    replay_dir: str | Path,
    save_dir: Optional[str | Path] = None,
    replay_count: int = 0,
) -> None:
    """
    Enable replay mode by creating a LiteLLMRecorder instance and monkey-patching litellm.completion.

    Args:
        replay_dir: Directory to load interactions from.
        save_dir: Directory to save new interactions. Defaults to replay_dir if None.
        replay_count: Number of interactions to replay before making live calls.
    """
    global _rllm_instance

    if _rllm_instance is not None:
        # Already enabled, do nothing
        return

    _rllm_instance = LiteLLMRecorder(
        replay_dir=replay_dir,
        save_dir=save_dir,
        replay_count=replay_count,
    )

    def patched_completion(**kwargs):
        return _rllm_instance.completion(**kwargs)

    litellm.completion = patched_completion


class LiteLLMRecorder(LLMRecorder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.completion_function = litellm.completion

    def make_live_call(self, **kwargs) -> Any:
        return self.completion_function(**kwargs)

    def dict_to_model_response(
        self, dict_response: Dict[str, Any]
    ) -> litellm.ModelResponse:
        return litellm.ModelResponse(**dict_response)

    def model_response_to_dict(
        self, model_response: litellm.ModelResponse
    ) -> Dict[str, Any]:
        return model_response.model_dump()
