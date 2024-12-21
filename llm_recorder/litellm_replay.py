from typing import Optional, Any, Dict, List, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import litellm
import logging

logger = logging.getLogger(__name__)

# this is for monkey patching
_rllm_instance: Optional['ReplayLiteLLM'] = None

def enable_replay_mode(
    replay_dir: str | Path,
    save_dir: Optional[str | Path] = None,
    replay_count: int = 0,
) -> None:
    """
    Enable replay mode by creating a ReplayLiteLLM instance and monkey-patching litellm.completion.
    
    Args:
        replay_dir: Directory to load interactions from.
        save_dir: Directory to save new interactions. Defaults to replay_dir if None.
        replay_count: Number of interactions to replay before making live calls.
    """
    global _original_completion, _rllm_instance

    if _rllm_instance is not None:
        # Already enabled, do nothing
        return

    _rllm_instance = ReplayLiteLLM(
        replay_dir=replay_dir,
        save_dir=save_dir,
        replay_count=replay_count,
        completion_function=litellm.completion,
    )

    def patched_completion(*args, **kwargs):
        # Remove debug print
        return _rllm_instance.completion(**kwargs)

    # Monkey-patch the litellm.completion function
    litellm.completion = patched_completion


@dataclass
class LLMInteraction:
    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]

    def save_to_directory(self, directory: Path, index: int) -> None:
        """Save request and response as separate files in the given directory"""
        # Save request
        with open(directory / f"{index}.request.json", 'w') as f:
            json.dump(self.request, f, indent=2)
            
        # Save response
        with open(directory / f"{index}.response.json", 'w') as f:
            json.dump(self.response, f, indent=2)

    @classmethod
    def load_from_directory(cls, directory: Path, index: int) -> 'LLMInteraction':
        """Load request and response from separate files"""
        # Load request
        with open(directory / f"{index}.request.json") as f:
            request = json.load(f)
            
        # Load response
        with open(directory / f"{index}.response.json") as f:
            response = json.load(f)
            
        return cls(
            timestamp=datetime.now().isoformat(),
            request=request,
            response=response
        )

class ReplayLiteLLM:
    def __init__(
        self,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
        completion_function: Callable = litellm.completion,
    ):
        """
        Initialize ReplayLiteLLM.
        
        Args:
            replay_dir: Directory to load interactions from
            save_dir: Optional directory to save new interactions. If None, saves to replay_dir
            replay_count: Number of interactions to replay.
                        After these are exhausted, live LLM calls will be made.
        """
        self.replay_dir = Path(replay_dir)
        if replay_count > 0 and not self.replay_dir.exists():
            raise FileNotFoundError(f"Replay directory not found: {self.replay_dir}")
        if self.replay_dir.exists() and not self.replay_dir.is_dir():
            raise ValueError(f"replay_dir must be a directory: {self.replay_dir}")
        
        self.save_dir = Path(save_dir) if save_dir else self.replay_dir
        if self.save_dir.exists() and not self.save_dir.is_dir():
            raise ValueError(f"save_dir must be a directory: {self.save_dir}")
        
        # Create save_dir if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.replay_count = replay_count
        self.replay_index = 0
        self.interactions: List[LLMInteraction] = []
        
        self._load_interactions()
        if replay_count > len(self.interactions):
            raise ValueError(f"replay_count is greater than the number of interactions in replay_dir: {len(self.interactions)}")

        for file in self.save_dir.glob("*.json"):
            file.unlink()
        
        self.completion_function = completion_function
        
    def _load_interactions(self) -> None:
        """Load interactions from replay directory"""
        # Find all request files and sort them
        request_files = sorted(self.replay_dir.glob("*.request.json"))
        
        # Load specified number of interactions
        for i, _ in enumerate(request_files[:self.replay_count], 1):
            try:
                interaction = LLMInteraction.load_from_directory(self.replay_dir, i)
                self.interactions.append(interaction)
            except FileNotFoundError as e:
                logger.warning(f"Incomplete interaction {i}: {e}")
                
    def _save_interaction(self, interaction: LLMInteraction) -> None:
        """Save an interaction to save directory"""
        # Find the next available index
        existing_files = list(self.save_dir.glob("*.request.json"))
        next_index = len(existing_files) + 1
        
        interaction.save_to_directory(self.save_dir, next_index)

    def _make_live_call(self, **kwargs) -> Any:
        """Make a live API call and save the interaction"""
        response = self.completion_function(**kwargs)
        
        interaction = LLMInteraction(
            timestamp=datetime.now().isoformat(),
            request=kwargs,
            response=self.model_response_to_dict(response)
        )
        return interaction

    def completion(self, **kwargs) -> Any:
        if self.replay_index < len(self.interactions):
            # If we have replay interactions available, use them
            interaction = self.interactions[self.replay_index]
            self.replay_index += 1
            logger.info(f"Replaying interaction {self.replay_index} of {len(self.interactions)}")
        else:
            # Otherwise make a live call
            interaction = self._make_live_call(**kwargs)
            logger.info("Making live LLM call")
        self._save_interaction(interaction)
        return self.dict_to_model_response(interaction.response)

    def dict_to_model_response(self, dict: Dict[str, Any]) -> litellm.ModelResponse:
        return litellm.ModelResponse(**dict)

    def model_response_to_dict(self, model_response: litellm.ModelResponse) -> Dict[str, Any]:
        return model_response.model_dump()


