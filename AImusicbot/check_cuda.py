import torch
from llama_cpp import Llama
import os

print("Checking CUDA availability...")
if torch.cuda.is_available():
    print("CUDA is available.")
    try:
        # Attempt to initialize Llama with GPU layers
        # This requires a dummy model path, but we won't actually load it.
        # The initialization itself might raise an error if CUDA is not properly linked.
        # We set verbose=False to avoid excessive output during initialization attempt.
        print("Attempting to initialize Llama with GPU layers...")
        # Use a non-existent path, as we only care if the CUDA part of initialization works
        dummy_model_path = "/tmp/dummy_model.gguf"
        # Ensure the dummy file exists, as Llama constructor checks file existence
        with open(dummy_model_path, 'w') as f:
            f.write("dummy content")

        # Initialize Llama with n_gpu_layers > 0
        # This will trigger the CUDA check within llama-cpp-python
        llm = Llama(model_path=dummy_model_path, n_gpu_layers=1, verbose=False)
        print("Llama initialized successfully with GPU layers. CUDA is likely working with llama-cpp-python.")
        del llm # Clean up the Llama instance

        # Clean up the dummy file
        os.remove(dummy_model_path)

    except Exception as e:
        print(f"Error initializing Llama with GPU layers: {e}")
        print("Llama-cpp-python might not be able to use CUDA despite torch reporting it as available.")
else:
    print("CUDA is not available according to torch.")
    print("Llama-cpp-python will not be able to use CUDA.")