from llm_recorder.llm_proxy import run_proxy
from anthropic import Anthropic
import os
import threading
import urllib.parse

MODEL = "claude-3-haiku-20240307"

# Get the original base URL from Anthropic
original_client = Anthropic()
original_base_url = original_client.base_url

# Encode the original base URL in the proxy URL
encoded_base = urllib.parse.quote(str(original_base_url).encode('utf-8'))
proxy_url = f"http://localhost:8000/{encoded_base}"

client = Anthropic(
    base_url=proxy_url
)

# Make some API calls
for i in range(3):
    print(f"\nCall {i + 1}:")

    message = client.messages.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Give me a one-sentence story about a cat (response #{i + 1})",
            }
        ],
        max_tokens=128,
        temperature=0.7,
        stream=False
    )

    print(f"Content: {message.content}")
    print(f"Model: {message.model}")
    print(f"Finish reason: {message.stop_reason}")
