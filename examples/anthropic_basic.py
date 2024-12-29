from llm_recorder.providers.anthropic_recorder import ReplayAnthropic
import os

MODEL = "claude-3-haiku-20240307"

client = ReplayAnthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    store_path="examples/saves/anthropic",
    replay_count=2,
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
    )

    print(f"Content: {message.content}")
    print(f"Model: {message.model}")
    print(f"Finish reason: {message.stop_reason}")
