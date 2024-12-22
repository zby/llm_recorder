# llm_recorder

`llm_recorder` is a basic observability and debugging tool for language model apps.
It stores interactions with LLMs in a local directory and provides a way to replay them.
It is particularly useful for debugging chained LLM calls.

With `llm_recorder`, you can:

- Save every request/response pair to a directory.
- Replay part of previously recorded interactions instead of making live LLM calls.

You can use it to:
- Inspect LLM responses and requests that lead to them
- Record an execution path that leads to a specific response and then replay it and see how the application behaves downstream
- Modify the recorded responses before replaying them

## Installation

```bash
pip install llm_recorder
```

## Getting Started

For a quick start use `enable_replay_mode`; it monkey patches `litellm` to record and replay responses.

By default `llm_recorder` records any calls to `litellm.completion` and their responses and stores them in the specified `save_dir`.
If no `save_dir` is specified, it will use the `replay_dir`.

Specify `replay_count` to replay previously recorded interactions.
Once the replayed interactions are exhausted, `llm_recorder` falls back to live LLM calls.

The `save_dir` is cleaned up at the start of each run, but `replay_dir` is read before that cleanup
so if they are the same directory, you can still replay the old interactions.


### Basic Example

```python
from llm_recorder import enable_replay_mode

import litellm


enable_replay_mode(replay_dir="saves")

print("After:", litellm.completion.__name__)

# Now just use litellm as usual:

response = litellm.completion(
    model="openai/gpt-4o",
    messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```
Then have a look at the 'saves' directory to see the recorded interactions.

### Chained Calls Example
See [examples/chained_calls.py](examples/chained_calls.py) for an example of how to use `llm_recorder` to record and replay chained LLM calls.
I saved the first response in [examples/saves/1.response.json](examples/saves/1.response.json) and the example
replays it before making another live call.
The saved response was from OpenAI, but the second call is directed to Anthropic.

## Advanced Usage

For more fine-grained control instantiate and use the LiteLLMRecorder class directly.
See [examples/direct_recorder.py](examples/direct_recorder.py).

## Limitations

- Currently only works with litellm. The support of direct use with OpenAI (see [examples/openai_basic.py](examples/openai_basic.py)) is experimental.
- Only supports synchronous calls.

## Contributing

Contributions are welcome! Please open issues or pull requests on GitHub.

## License

This project is licensed under the MIT License.


