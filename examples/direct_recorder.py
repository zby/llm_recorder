from pathlib import Path
import litellm
from llm_recorder.providers.litellm_recorder import LitellmRecorder

MODEL = "gpt-4o-mini"
MODEL = "anthropic/claude-3-5-haiku-latest"


def main():
    # Initialize the recorder
    recorder = LitellmRecorder(
        store_path="examples/saves/direct",
        replay_count=2,  # Set to 0 to make live calls, or N to replay N recordings
    )

    # First call
    response1 = recorder.completion(
        model=MODEL,
        messages=[{"role": "user", "content": "Write a haiku about coding. Please don't explain it."}],
    )
    print("\nFirst response:")
    print(response1.choices[0].message.content)

    # Use the response in a follow-up call
    follow_up = f"Write a short explanation of this haiku: {response1.choices[0].message.content}"
    response2 = recorder.completion(
        model=MODEL, messages=[{"role": "user", "content": follow_up}]
    )
    print("\nSecond response:")
    print(response2.choices[0].message.content)

    # Make a third call incorporating both previous responses
    summary_prompt = (
        f"Summarize these two interactions:\n"
        f"1. Haiku: {response1.choices[0].message.content}\n"
        f"2. Explanation: {response2.choices[0].message.content}"
    )
    response3 = recorder.completion(
        model=MODEL, messages=[{"role": "user", "content": summary_prompt}]
    )
    print("\nThird response (summary):")
    print(response3.choices[0].message.content)


if __name__ == "__main__":
    # Make sure the directories exist

    main()
