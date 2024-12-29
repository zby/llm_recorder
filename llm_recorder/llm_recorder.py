import logging
from typing import Any, List, Optional, Union, Dict, Protocol
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


@dataclass
class LLMInteraction:
    timestamp: str
    request: Dict[str, Any]
    response: Dict[str, Any]


class Persistence(Protocol):
    def load_all(self, limit: int) -> List[LLMInteraction]:
        ...
    def save(self, interaction: LLMInteraction) -> None:
        ...


class FilePersistence:
    """
    A file-based persistence layer that saves each LLMInteraction
    in separate JSON files: 
      - 1.request.json
      - 1.response.json
      - etc.
    """
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._current_index = 0

    def load_all(self, limit: int) -> List[LLMInteraction]:
        interactions = []
        # We rely on files named "1.request_*.json", "2.request_*.json", etc.
        request_files = sorted(self.directory.glob("*.request_*.json"))

        # Get unique indices from request files
        unique_indices = {int(f.name.split('.')[0]) for f in request_files}
        sorted_indices = sorted(unique_indices)

        for i in sorted_indices[:limit]:
            try:
                interaction = self._load_single_interaction(i)
                interactions.append(interaction)
            except FileNotFoundError as e:
                logger.warning(f"Incomplete interaction at index={i}: {e}")

        # Clean up the directory after loading
        self._cleanup_directory()
        self._current_index = 0
        return interactions

    def _cleanup_directory(self) -> None:
        """Remove all existing files in the directory."""
        for file in self.directory.glob("*.json"):
            file.unlink()

    def save(self, interaction: LLMInteraction) -> None:
        self._current_index += 1

        # Save each key in the request dictionary as a separate file
        for key, value in interaction.request.items():
            filename = f"{self._current_index}.request_{key}.json"
            with open(self.directory / filename, "w") as f:
                json.dump(value, f, indent=2)

        # Save each key in the response dictionary as a separate file
        for key, value in interaction.response.items():
            filename = f"{self._current_index}.response_{key}.json"
            with open(self.directory / filename, "w") as f:
                json.dump(value, f, indent=2)

    def _load_single_interaction(self, index: int) -> LLMInteraction:
        # Find all request and response files for this index
        request_files = self.directory.glob(f"{index}.request_*.json")
        response_files = self.directory.glob(f"{index}.response_*.json")

        # Build request dictionary
        request_data = {}
        for file in request_files:
            # First get the part after 'request_'
            after_prefix = str(file).split('request_')[1]
            # Then remove the '.json' suffix
            key = after_prefix.rsplit('.json', 1)[0]
            with file.open("r") as f:
                request_data[key] = json.load(f)

        # Build response dictionary
        response_data = {}
        for file in response_files:
            # First get the part after 'response_'
            after_prefix = str(file).split('response_')[1]
            # Then remove the '.json' suffix
            key = after_prefix.rsplit('.json', 1)[0]
            with file.open("r") as f:
                response_data[key] = json.load(f)

        return LLMInteraction(
            timestamp="",
            request=request_data,
            response=response_data
        )

class LLMRecorder(ABC):
    """
    A simple recorder that automatically creates a FilePersistence by default,
    unless a custom persistence is provided.
    """

    def __init__(
        self,
        store_path: Union[str, Path],
        replay_count: int = 0,
        persistence: Optional["Persistence"] = None
    ):
        """
        Initialize an LLMRecorder.

        If no `persistence` object is provided, a default FilePersistence
        will be created using `store_path`.

        Args:
            live_call: Callable that performs the actual API/LLM call
            store_path: Directory where interactions are stored
            replay_count: The number of interactions to replay before making live calls, defaults to 0
            persistence: (Optional) a custom Persistence implementation. If provided,
                         it overrides the default FilePersistence creation.
        """

        # If the user didn't provide a persistence object, create a FilePersistence
        if persistence is None:
            self.persistence = FilePersistence(directory=Path(store_path))
        else:
            self.persistence = persistence

        self.replay_count = replay_count
        self.replay_index = 0

        # Load existing interactions (up to replay_count) from the persistence layer
        self.interactions: List[LLMInteraction] = self.persistence.load_all(limit=self.replay_count)
        if replay_count > len(self.interactions):
            raise ValueError(
                f"replay_count ({replay_count}) > available interactions ({len(self.interactions)})"
            )

    @abstractmethod
    def live_call(self, **kwargs) -> Any:
        """
        Make a live call to the LLM.
        """
        pass


    @abstractmethod
    def req_to_dict(self, req: Any) -> Dict[str, Any]:
        """
        Convert a request to a dictionary.
        """
        pass

    @abstractmethod
    def res_to_dict(self, res: Any) -> Dict[str, Any]:
        """
        Convert a response to a dictionary.
        """
        pass

    def dict_completion(self, **kwargs) -> Dict[str, Any] :
        """
        Either replay a saved interaction or make a new call (which is recorded).
        """
        # If we have replay interactions left, replay them
        if self.replay_index < len(self.interactions):
            interaction = self.interactions[self.replay_index]
            self.replay_index += 1
            logger.info(f"Replaying interaction #{self.replay_index}")
        else:
            # Otherwise, make a live call and create a new interaction
            logger.info("Making live call (no more replays available)")
            response = self.live_call(**kwargs)
            interaction = LLMInteraction(
                timestamp=datetime.now().isoformat(),
                request=self.req_to_dict(kwargs),
                response=self.res_to_dict(response),
            )
            # Save the new interaction
            self.persistence.save(interaction)

        return interaction.response

if __name__ == "__main__":

    class ExampleLLMRecorder(LLMRecorder):
        def live_call(self, **kwargs) -> Dict[str, Any]:
            return {"answer": f"Echo: {kwargs.get('prompt', '')}"}

        def req_to_dict(self, req: Any) -> Dict[str, Any]:
            return req

        def res_to_dict(self, res: Any) -> Dict[str, Any]:
            return res

        def completion(self, **kwargs) -> Dict[str, Any]:
            return self.dict_completion(**kwargs)

    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)

    # Example 1: Recording new interactions
    recorder = ExampleLLMRecorder(
        store_path="./example_logs"
    )

    # Make some example calls that will be recorded
    result1 = recorder.completion(
        prompt="What is the capital of France?",
        temperature=0.7,
        headers={"Authorization": "Bearer sk-..."}
    )
    print("First call result:", result1)

    result2 = recorder.completion(
        prompt="What is the capital of Japan?",
        temperature=0.7,
        headers={"Authorization": "Bearer sk-..."}
    )
    print("Second call result:", result2)

    # Example 2: Replaying recorded interactions
    replay_recorder = ExampleLLMRecorder(
        store_path="./example_logs",
        replay_count=2  # Replay the two interactions we just recorded
    )

    # These calls will use the recorded responses instead of making live calls
    replay1 = replay_recorder.completion(
        prompt="This prompt will be ignored because we're replaying",
        temperature=0.5
    )
    print("First replay:", replay1)

    replay2 = replay_recorder.completion(
        prompt="This prompt will also be ignored",
        temperature=0.5
    )
    print("Second replay:", replay2)

    # After replay_count is exhausted, this will make a live call
    live_call = replay_recorder.completion(
        prompt="This will make a live call",
        temperature=0.5
    )
    print("Live call after replays:", live_call)
