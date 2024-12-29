from typing import Dict, Any
from pathlib import Path
from ..llm_recorder import LLMRecorder

try:
    from google.generativeai.types import GenerateContentResponse
    from google.generativeai.generative_models import GenerativeModel
    import google.generativeai.protos as protos
except ImportError:
    raise ImportError(
        "Google provider requires the 'google-generativeai' package. "
        "Install it with 'pip install llm-recorder[google]'"
    )


class RecorderGenerativeModel(GenerativeModel, LLMRecorder):
    """Subclass of GenerativeModel that supports recording/replaying interactions"""

    def __init__(
        self,
        model_name: str,
        store_path: str | Path,
        replay_count: int = 0,
        **kwargs,
    ):
        """
        Initialize RecorderGenerativeModel.

        Args:
            model_name: Name of the Google model to use (e.g., "gemini-pro")
            store_path: Directory to load interactions from
            replay_count: Number of interactions to replay before making live calls
            **kwargs: Additional arguments passed to GenerativeModel constructor
        """
        super().__init__(model_name=model_name, **kwargs)
        LLMRecorder.__init__(
            self,
            store_path=store_path,
            replay_count=replay_count,
        )

    def live_call(self, **kwargs) -> GenerateContentResponse:
        """Make a live API call to Google"""
        response = super().generate_content(**kwargs)
        return response
    
    def req_to_dict(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a request to a dictionary"""
        return req

    def res_to_dict(self, res: GenerateContentResponse) -> Dict[str, Any]:
        """Convert a response to a dictionary"""
        return res.to_dict()

    def generate_content(self, contents: str, **kwargs) -> GenerateContentResponse:
        """Generate content with recording/replay support"""
        kwargs["contents"] = contents

        dict_response = self.dict_completion(**kwargs)
        """Convert a dictionary back to a model response object"""
        response = protos.GenerateContentResponse(**dict_response)
        return GenerateContentResponse.from_response(response)

