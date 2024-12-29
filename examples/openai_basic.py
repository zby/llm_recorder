from llm_recorder.providers.openai_recorder import OpenAIRecorder

# Note: This example assumes that you have set OPENAI_API_KEY in your environment.

client = OpenAIRecorder(
    store_path="examples/saves/openai",
    replay_count=0,  # Will replay first 2 interactions, then make live calls
)
# client is now a replacement for the OpenAI client

# Make some API calls
for i in range(3):
    print(f"\nCall {i + 1}:")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Give me a one-sentence story about a cat (response #{i + 1})",
            }
        ],
        temperature=0.7,
    )

    print(f"Content: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
