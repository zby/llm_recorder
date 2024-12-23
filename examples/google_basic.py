import os
import google.generativeai as genai
from llm_recorder.providers.google_recorder import RecorderGenerativeModel

# Configure the Google API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Get a model with recording capabilities
model = RecorderGenerativeModel(
    "gemini-1.5-flash",
    replay_dir="examples/saves/google",
    save_dir="examples/saves/google",
    replay_count=2,
)

# Make some API calls
for i in range(3):
    print(f"\nCall {i + 1}:")

    response = model.generate_content(
        "Give me a one-sentence story about a cat",
    )

    print(f"Content: {response.text}")
    print(f"Model: {model.model_name}")
    print(f"Finish reason: {response.candidates[0].finish_reason}")
