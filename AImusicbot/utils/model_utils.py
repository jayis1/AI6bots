import logging
import os
import requests

class AIError(Exception):
    """Custom exception for AI-related errors."""
    pass

def download_model(model_url: str, model_name: str, model_dir: str = "LizBotz-v2.0/models/"):
    """
    Downloads an AI model from a given URL and saves it to the specified directory.

    Args:
        model_url (str): The URL of the model file to download.
        model_name (str): The name to save the model file as.
        model_dir (str): The directory to save the model in. Defaults to "LizBotz-v2.0/models/".
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    logging.info(f"Attempting to download model from {model_url} to {model_path}")

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Successfully downloaded model to {model_path}")
        return model_path

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading model from {model_url}: {e}", exc_info=True)
        raise AIError(f"Failed to download model: {e}")
    except IOError as e:
        logging.error(f"Error saving model to {model_path}: {e}", exc_info=True)
        raise AIError(f"Failed to save model file: {e}")
