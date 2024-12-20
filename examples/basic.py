from llm_recorder import enable_replay_mode

import litellm


enable_replay_mode(replay_dir="saves")

print("After patching:", litellm.completion.__name__)

# Now just use litellm as usual:

response = litellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)

# Now the request and response are recorded in the saves directory