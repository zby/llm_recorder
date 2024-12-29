from llm_recorder import enable_replay_mode

import litellm

# Note: This example assumes that you have set up your API credentials
# in your environment variables. Depending on the model you choose, you'll need to set:
# - OPENAI_API_KEY for OpenAI models
# - ANTHROPIC_API_KEY for Anthropic models


enable_replay_mode(store_path="examples/saves/basic")

print("After patching:", litellm.completion.__name__)

# Now just use litellm as usual:

response = litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"content": "Hello, how are you?", "role": "user"}],
)

print(response)
# print(response.choices[0].message.content)

# Now the request and response are recorded in the examples/saves/basic directory
