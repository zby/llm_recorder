from typing import Dict, Any, Optional
from pathlib import Path
from ..llm_recorder import LLMRecorder

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    from google.generativeai.generative_models import GenerativeModel
    import google.generativeai.protos as protos
except ImportError:
    raise ImportError(
        "Google provider requires the 'google-generativeai' package. "
        "Install it with 'pip install llm-recorder[google]'"
    )


class GoogleLLMRecorder(LLMRecorder):
    """Implementation of LLMRecorder for Google's Generative AI"""

    def __init__(self, original_generate_content, **kwargs):
        super().__init__(**kwargs)
        self.completion_arg_names = ["contents"]
        self.original_generate_content = original_generate_content

    def make_live_call(self, *args, **kwargs) -> GenerateContentResponse:
        """Make a live API call to Google"""
        response = self.original_generate_content(*args, **kwargs)
        return response

    def dict_to_model_response(
        self, dict_response: Dict[str, Any]
    ) -> GenerateContentResponse:
        """Convert a dictionary back to a model response object"""
        response = protos.GenerateContentResponse(**dict_response)
        return GenerateContentResponse.from_response(response)

    def model_response_to_dict(
        self, model_response: GenerateContentResponse
    ) -> Dict[str, Any]:
        """Convert a model response object to a dictionary"""
        return model_response.to_dict()


class RecorderGenerativeModel(GenerativeModel):
    """Subclass of GenerativeModel that supports recording/replaying interactions"""

    def __init__(
        self,
        model_name: str,
        replay_dir: str | Path,
        save_dir: Optional[str | Path] = None,
        replay_count: int = 0,
        **kwargs,
    ):
        """
        Initialize RecorderGenerativeModel.

        Args:
            model_name: Name of the Google model to use (e.g., "gemini-pro")
            replay_dir: Directory to load interactions from
            save_dir: Optional directory to save new interactions. If None, saves to replay_dir
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to GenerativeModel constructor
        """
        super().__init__(model_name=model_name, **kwargs)
        original_generate_content = super().generate_content
        self._recorder = GoogleLLMRecorder(
            original_generate_content=original_generate_content,
            replay_dir=replay_dir,
            save_dir=save_dir,
            replay_count=replay_count,
        )

    def generate_content(self, *args, **kwargs) -> GenerateContentResponse:
        """Generate content with recording/replay support"""
        return self._recorder.completion(*args, **kwargs)
