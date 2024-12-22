from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class LLMInteraction:
    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]

    def save_to_directory(self, directory: Path, index: int) -> None:
        """Save request and response as separate files in the given directory"""
        with open(directory / f"{index}.request.json", 'w') as f:
            json.dump(self.request, f, indent=2)
            
        with open(directory / f"{index}.response.json", 'w') as f:
            json.dump(self.response, f, indent=2)

    @classmethod
    def load_from_directory(cls, directory: Path, index: int) -> 'LLMInteraction':
        """Load request and response from separate files"""
        with open(directory / f"{index}.request.json") as f:
            request = json.load(f)
            
        with open(directory / f"{index}.response.json") as f:
            response = json.load(f)
            
        return cls(
            timestamp=datetime.now().isoformat(),
            request=request,
            response=response
        )

class LLMRecorder(ABC):
    def __init__(
        self,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
    ):
        """
        Initialize ReplayLLM.
        
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

    def _load_interactions(self) -> None:
        """Load interactions from replay directory"""
        request_files = sorted(self.replay_dir.glob("*.request.json"))
        
        for i, _ in enumerate(request_files[:self.replay_count], 1):
            try:
                interaction = LLMInteraction.load_from_directory(self.replay_dir, i)
                self.interactions.append(interaction)
            except FileNotFoundError as e:
                logger.warning(f"Incomplete interaction {i}: {e}")
                
    def _save_interaction(self, interaction: LLMInteraction) -> None:
        """Save an interaction to save directory"""
        existing_files = list(self.save_dir.glob("*.request.json"))
        next_index = len(existing_files) + 1
        
        interaction.save_to_directory(self.save_dir, next_index)

    @abstractmethod
    def make_live_call(self, **kwargs) -> Any:
        """Make a live API call"""
        pass

    @abstractmethod
    def dict_to_model_response(self, dict_response: Dict[str, Any]) -> Any:
        """Convert a dictionary back to a model response object"""
        pass

    @abstractmethod
    def model_response_to_dict(self, model_response: Any) -> Dict[str, Any]:
        """Convert a model response object to a dictionary"""
        pass

    def completion(self, **kwargs) -> Any:
        if self.replay_index < len(self.interactions):
            # If we have replay interactions available, use them
            interaction = self.interactions[self.replay_index]
            self.replay_index += 1
            logger.info(f"Replaying interaction {self.replay_index} of {len(self.interactions)}")
        else:
            # Otherwise make a live call
            response = self.make_live_call(**kwargs)
            interaction = LLMInteraction(
                timestamp=datetime.now().isoformat(),
                request=kwargs,
                response=self.model_response_to_dict(response)
            )
            logger.info("Making live LLM call")
        
        self._save_interaction(interaction)
        return self.dict_to_model_response(interaction.response) 