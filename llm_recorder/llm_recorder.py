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
    def load_all(self, limit: int) -> List[LLMInteraction]: ...
    def save(self, interaction: LLMInteraction) -> None: ...


class FilePersistence:
    """
    A file-based persistence layer that saves each LLMInteraction
    in separate JSON files:
      - 1.request_xxx.json
      - 1.response_xxx.json
      - etc.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.loaded_files = set()

    def load_all(self, limit: int) -> List[LLMInteraction]:
        interactions = []
        # We rely on files named "1.request_*.json", "2.request_*.json", etc.
        request_files = sorted(self.directory.glob("*.request_*.json"))

        # Get unique indices from request files
        unique_indices = {int(f.name.split(".")[0]) for f in request_files}
        sorted_indices = sorted(unique_indices)

        for i in sorted_indices[:limit]:
            try:
                interaction = self._load_single_interaction(i)
                interactions.append(interaction)
            except FileNotFoundError as e:
                logger.warning(f"Incomplete interaction at index={i}: {e}")

        # Clean up the directory after loading
        self._cleanup_directory()
        return interactions

    def _cleanup_directory(self) -> None:
        """Remove files that weren't loaded."""
        for file in self.directory.glob("*.json"):
            if file not in self.loaded_files:
                file.unlink()

    def save(self, interaction: LLMInteraction, index: int) -> None:
        # Save each key in the request dictionary as a separate file
        for key, value in interaction.request.items():
            filename = f"{index}.request_{key}.json"
            with open(self.directory / filename, "w") as f:
                json.dump(value, f, indent=2)

        # Save each key in the response dictionary as a separate file
        for key, value in interaction.response.items():
            filename = f"{index}.response_{key}.json"
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
            after_prefix = str(file).split("request_")[1]
            # Then remove the '.json' suffix
            key = after_prefix.rsplit(".json", 1)[0]
            with file.open("r") as f:
                request_data[key] = json.load(f)
            self.loaded_files.add(file)  # Track this file as loaded

        # Build response dictionary
        response_data = {}
        for file in response_files:
            # First get the part after 'response_'
            after_prefix = str(file).split("response_")[1]
            # Then remove the '.json' suffix
            key = after_prefix.rsplit(".json", 1)[0]
            with file.open("r") as f:
                response_data[key] = json.load(f)
            self.loaded_files.add(file)  # Track this file as loaded

        return LLMInteraction(
            timestamp="", request=request_data, response=response_data
        )


class LLMRecorder(ABC):
    """
    A concrete subclass must implement the following methods:
    - live_call(**kwargs): Make actual calls to the LLM API
    - req_to_dict(req): Convert request parameters to a serializable dictionary
    - res_to_dict(res): Convert API response to a serializable dictionary
    """

    def __init__(
        self,
        persistence: Union[str, Path, "Persistence"],
        replay_count: int = 0,
    ):
        """
        Initialize an LLMRecorder.

        Args:
            persistence: Either a Persistence implementation or a path (str/Path) for default FilePersistence
            replay_count: The number of interactions to replay before making live calls, defaults to 0
        """
        # If persistence is a string/Path, create a FilePersistence
        if isinstance(persistence, (str, Path)):
            self.persistence = FilePersistence(directory=Path(persistence))
        else:
            self.persistence = persistence

        self.replay_count = replay_count
        self.replay_index = 0

        # Load existing interactions (up to replay_count) from the persistence layer
        self.interactions: List[LLMInteraction] = self.persistence.load_all(
            limit=self.replay_count
        )
        if replay_count > len(self.interactions):
            raise ValueError(
                f"Cannot replay ({replay_count}) interactions - there are only ({len(self.interactions)}) available"
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

    def _replay_interaction(self) -> LLMInteraction:
        """Replay a saved interaction."""
        interaction = self.interactions[self.replay_index]
        logger.info(f"Replaying interaction #{self.replay_index}")
        return interaction

    def _make_live_call(self, **kwargs) -> LLMInteraction:
        """Make a live call and create a new interaction."""
        logger.info("Making live call (no more replays available)")
        response = self.live_call(**kwargs)
        interaction = LLMInteraction(
            timestamp=datetime.now().isoformat(),
            request=self.req_to_dict(kwargs),
            response=self.res_to_dict(response),
        )
        # Save the new interaction immediately
        self.persistence.save(interaction, self.replay_index + 1)
        return interaction

    def dict_completion(self, **kwargs) -> Dict[str, Any]:
        """
        Either replay a saved interaction or make a new call (which is recorded).
        """
        # If we have replay interactions left, replay them
        if self.replay_index < len(self.interactions):
            interaction = self._replay_interaction()
        else:
            # Otherwise, make a live call and create a new interaction
            interaction = self._make_live_call(**kwargs)

        self.replay_index += 1
        return interaction.response


if __name__ == "__main__":

    class ExampleCompletion:
        def __init__(self, answer: str):
            self.answer = answer
            self.count = 0

        def completion(self, **kwargs) -> Dict[str, Any]:
            self.count += 1
            return {"answer": f"{self.answer} {self.count}"}

    class ExampleLLMRecorder(ExampleCompletion, LLMRecorder):

        def __init__(self, persistence: Union[str, Path, "Persistence"], replay_count: int = 0):
            LLMRecorder.__init__(self, persistence=persistence, replay_count=replay_count)
            ExampleCompletion.__init__(self, answer="Test")

        def live_call(self, **kwargs) -> Dict[str, Any]:
            return super().completion(**kwargs)

        def req_to_dict(self, req: Any) -> Dict[str, Any]:
            return req

        def res_to_dict(self, res: Any) -> Dict[str, Any]:
            return res

        def completion(self, **kwargs) -> Dict[str, Any]:
            # override the completion method so that it uses the reload logic
            return self.dict_completion(**kwargs)

    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)

    # Example 1: Recording new interactions
    recorder = ExampleLLMRecorder("./example_logs")

    # Make some example calls that will be recorded
    result1 = recorder.completion(
        prompt="What is the capital of France?",
        temperature=0.7,
        headers={"Authorization": "Bearer sk-..."},
    )
    print("First call result:", result1)

    result2 = recorder.completion(
        prompt="What is the capital of Japan?",
        temperature=0.7,
        headers={"Authorization": "Bearer sk-..."},
    )
    print("Second call result:", result2)

    # Example 2: Replaying recorded interactions
    replay_recorder = ExampleLLMRecorder(
        persistence="./example_logs",
        replay_count=2,  # Replay the two interactions we just recorded
    )

    # These calls will use the recorded responses instead of making live calls
    replay1 = replay_recorder.completion(
        prompt="This prompt will be ignored because we're replaying", temperature=0.5
    )
    print("First replay:", replay1)

    replay2 = replay_recorder.completion(
        prompt="This prompt will also be ignored", temperature=0.5
    )
    print("Second replay:", replay2)

    # After replay_count is exhausted, this will make a live call
    live_call = replay_recorder.completion(
        prompt="This will make a live call", temperature=0.5
    )
    print("Live call after replays:", live_call)
