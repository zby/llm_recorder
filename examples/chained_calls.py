import os
from llm_recorder import enable_replay_mode
import litellm

#MODEL = "openai/gpt-4o-mini"
MODEL = "anthropic/claude-3-5-sonnet-20240620"

script_dir = os.path.dirname(os.path.abspath(__file__))
saves_dir = os.path.join(script_dir, "saves")

enable_replay_mode(replay_dir=saves_dir, replay_count=1)
# replay_count=1 means that the first call will be replayed and the second will be live
# there is one request response pair saved in the saves directory - it was generated with the OpenAI model
# now we can replay it and then make another live call to Anthropic

# System message to set the context for our interaction
system_message = {
    "role": "system",
    "content": """You are a knowledgeable teacher who first suggests interesting topics to learn about,
then provides short explanations about the chosen topic. Please keep the explanations to 3 sentences or less."""
}

# First request: Get a topic suggestion
first_response = litellm.completion(
    model=MODEL,
    messages=[
        system_message,
        {
            "role": "user",
            "content": "Suggest an interesting scientific topic that most people don't know about. Keep it to one sentence."
        }
    ]
)

# Extract the topic from the first response
suggested_topic = first_response.choices[0].message.content

print("Suggested topic:", suggested_topic)

# Second request: Get detailed information about the topic
second_response = litellm.completion(
    model=MODEL,
    messages=[
        system_message,
        {
            "role": "user",
            "content": "Suggest an interesting scientific topic that most people don't know about. Keep it to one sentence."
        },
        {
            "role": "assistant",
            "content": suggested_topic
        },
        {
            "role": "user",
            "content": f"That's interesting! Can you explain {suggested_topic} in more detail? Give me 3 fascinating facts about it."
        }
    ]
)

print("\nDetailed explanation:")
print(second_response.choices[0].message.content)

# Both requests and responses will be saved in the ./saves directory 