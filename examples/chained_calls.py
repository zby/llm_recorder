from llm_recorder import enable_replay_mode
import litellm

# This is a simple example of how to use llm_recorder to record and replay chained LLM calls.
# There is a saved response in the saves directory from the first call.
# Now we can replay it and then make another live call to Anthropic.


# Note: This example assumes that you have set up your API credentials
# in your environment variables. Depending on the model you choose, you'll need to set:
# - OPENAI_API_KEY for OpenAI models
# - ANTHROPIC_API_KEY for Anthropic models

MODEL = "anthropic/claude-3-5-haiku-latest"
#MODEL = "openai/gpt-4o-mini"


enable_replay_mode("examples/saves/chained_calls", replay_count=1)
# replay_count=1 means that the first call will be replayed and the second will be live

# System message to set the context for our interaction
system_message = {
    "role": "system",
    "content": """You are a knowledgeable teacher who first suggests interesting topics to learn about,
then provides short explanations about the chosen topic. Please keep the explanations to 3 sentences or less.""",
}

# First request: Get a topic suggestion
first_response = litellm.completion(
    model=MODEL,
    messages=[
        system_message,
        {
            "role": "user",
            "content": "Suggest an interesting scientific topic that most people don't know about. Keep it to one sentence.",
        },
    ],
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
            "content": "Suggest an interesting scientific topic that most people don't know about. Keep it to one sentence.",
        },
        {"role": "assistant", "content": suggested_topic},
        {
            "role": "user",
            "content": f"That's interesting! Can you explain {suggested_topic} in more detail? Give me 3 fascinating facts about it.",
        },
    ],
)

print("\nDetailed explanation:")
print(second_response.choices[0].message.content)

# Both requests and responses will be saved in the ./saves directory
