from typing import Any, Dict, Union
import httpx
from pathlib import Path
from llm_recorder import LLMRecorder
from openai import OpenAI
import json


class HTTPRecorder(httpx.Client, LLMRecorder):
    """
    An HTTP client that records and replays LLM API interactions.
    Inherits from both httpx.Client and LLMRecorder.
    """

    def __init__(self, store_path: Union[str, Path], replay_count: int = 0, **kwargs):
        # Initialize both parent classes
        httpx.Client.__init__(self, **kwargs)
        LLMRecorder.__init__(self, store_path=store_path, replay_count=replay_count)

    def live_call(self, **kwargs) -> httpx.Response:
        """Make a live HTTP request using the parent httpx.Client"""

        # Make the actual HTTP request
        return super().send(**kwargs)

    def res_to_dict(self, res: httpx.Response) -> Dict[str, Any]:
        """Convert an httpx.Response object to a dictionary"""
        return {
            "status_code": res.status_code,
            "headers": dict(res.headers),
            "json": res.json(),
        }

    def req_to_dict(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        request = kwargs.pop("request", None)
        """Convert an httpx.Request object to a dictionary"""
        content = request.content.decode("utf-8")
        decoded_content = json.loads(content)

        return {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": decoded_content,
            "kwargs": kwargs,
        }

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        """
        Override send method which is used by the OpenAI client.
        """
        kwargs["request"] = request
        dict_response = self.dict_completion(**kwargs)
        headers = dict_response.pop("headers")
        headers.pop("content-encoding")
        json_content = dict_response.pop("json")
        text = json.dumps(json_content)
        status_code = dict_response.pop("status_code")
        response = httpx.Response(
            status_code=status_code, headers=headers, request=request, text=text
        )
        return response


# Example usage
if __name__ == "__main__":
    # Create a recorder instance
    http_client = HTTPRecorder(
        store_path="./http_logs",
    )

    # Create OpenAI client with our recorder as the http_client
    client = OpenAI(http_client=http_client)

    # Make a request that will be recorded
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello!"}]
    )

    print(f"Response: {response.choices[0].message.content}")
