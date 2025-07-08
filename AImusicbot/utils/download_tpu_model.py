#!/usr/bin/env python3

import os
import logging
from utils.model_utils import download_model, AIError # Import download_model and AIError from model_utils

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')

# Define the expected path and a placeholder URL for the TFLite model
MODEL_DIR = "LizBotz-v2.0/models"
MODEL_NAME = "mobilebert_qa_squad_edgetpu.tflite" # Using a more specific placeholder filename
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
# IMPORTANT: Replace with the actual URL of your quantized TFLite model for Edge TPU from TensorFlow Hub
# Example URL pattern for a MobileBERT QA model on TF Hub
MODEL_URL = f"https://tfhub.dev/google/lite-model/mobilebert/lite/qa/squad/1/{MODEL_NAME}" # Constructed plausible URL

def main():
    """
    Checks if the TPU model exists and downloads it if missing.
    """
    logging.info(f"Checking for TPU model at {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        logging.warning(f"TPU model not found at {MODEL_PATH}. Attempting to download...")
        try:
            # Use the standalone download_model function
            downloaded_path = download_model(MODEL_URL, MODEL_NAME)
            logging.info(f"Download complete. Model saved to {downloaded_path}")
        except AIError as e:
            logging.error(f"Failed to download TPU model: {e}")
            # Exit with a non-zero status to indicate failure in the script
            exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred during model download: {e}", exc_info=True)
            exit(1)
    else:
        logging.info("TPU model already exists. Skipping download.")

if __name__ == "__main__":
    # Ensure the models directory exists before checking for the model file
    os.makedirs(MODEL_DIR, exist_ok=True)
    main()