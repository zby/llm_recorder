import os
from openai import OpenAI
from llm_recorder.openai_replay import openai_enable_replay_mode

# Note: This example assumes that you have set OPENAI_API_KEY in your environment.

# Create the OpenAI client
client = OpenAI()

# Enable replay mode
client = openai_enable_replay_mode(
    client=client,
    replay_dir="examples/saves_openai",
    replay_count=2  # Will replay first 2 interactions, then make live calls
)

# Make some API calls
for i in range(3):
    print(f"\nCall {i + 1}:")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Give me a one-sentence story about a cat (response #{i + 1})"
            }
        ],
        temperature=0.7,
    )
    
    print(f"Content: {response.choices[0].message.content}")
    print(f"Model: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}") 