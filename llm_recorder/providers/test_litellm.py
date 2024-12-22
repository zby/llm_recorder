import pytest
from pathlib import Path
import json
import tempfile
import litellm
from unittest.mock import patch, MagicMock
import llm_recorder.providers.litellm_recorder as litellm_recorder



def test_enable_replay_mode():
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Store original completion function
        original_completion = litellm.completion
        
        # Enable replay mode
        litellm_recorder.enable_replay_mode(temp_path)
        
        # Check that _rllm_instance was created and is the correct type
        assert litellm_recorder._rllm_instance is not None
        assert isinstance(litellm_recorder._rllm_instance, litellm_recorder.LiteLLMRecorder)
        
        # Check that litellm.completion has been changed
        assert litellm.completion != original_completion
        assert litellm.completion.__name__ == 'patched_completion'
