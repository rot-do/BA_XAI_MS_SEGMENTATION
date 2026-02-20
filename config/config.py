import json
import logging
import platform
import os
import importlib
from typing import Any, Dict, Optional

import torch

if not torch.cuda.is_available():
    try:
        import torch_directml
    except:
        print()
from config import logger as lg



def load_config() -> Dict[str, Any]:
    """
    Loads configuration from the config.json file based on the current machine name.

    Returns:
        Dict[str, Any]: Configuration dictionary for the specific PC.

    Raises:
        Exception: If the current PC name is not recognized.
    """
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
        config = json.load(f)

    current_pc = platform.node()

    if current_pc == "DESKTOP-9KJKDD5":
        return config["pc"]
    elif current_pc == "Dominik":
        return config["laptop"]
    elif current_pc == "DESKTOP-M0GAHBC":
        return config["lea"]
    else:
        error = f"Unknown PC: {current_pc}. This machine must be added to the configuration with the appropriate paths."
        raise Exception(error)


try:
    config: Dict[str, Any] = load_config()
    logs: str = config.get("logs", "logs")

    logger = lg.setup_logger('logger', log_dir=logs, level=logging.DEBUG)
    logger.info("Logger initialized")

    def safe_get(key: str) -> Optional[str]:
        value = config.get(key)
        if value is None:
            logger.warning(f"Key '{key}' not found in config, skipping.")
        return value

    brain_mri = safe_get("Brain_MRI_dataset")
    models = safe_get("models")
    msseg2016_train = safe_get("MSSEG2016_Train")
    msseg2016_test = safe_get("MSSEG2016_Test")
    msseg2016 = safe_get("MSSEG2016")

    mslesseg_train = safe_get("MSLESSEG_Train")
    mslesseg_test = safe_get("MSLESSEG_Test")
    mslesseg = safe_get("MSLESSEG")
    mslesseg_raw = safe_get("MSLESSEG_RAW")
    mslesseg_template = safe_get("MSLESSEG_TEMPLATE")

    output_path = safe_get("output_path")
    nnUNet_raw = safe_get("nnunet_raw_data")
    nnUNet_preprocessed = safe_get("nnunet_preprocessed")
    nnunet_predictions = safe_get("nnunet_predictions")

    dataset_301 = safe_get("dataset301")
    dataset_304 = safe_get("dataset304")
    dataset_305 = safe_get("dataset305")
    dataset_306 = safe_get("dataset306")

    dataset_200 = safe_get("dataset200")
    dataset_201 = safe_get("dataset201")
    dataset_210 = safe_get("dataset210")
    dataset_211 = safe_get("dataset211")

    nnunet_trained_models = safe_get("nnunet_trained_models")
    nnunet_trained_model_2d_301 = safe_get("nnunet_trained_model_2d_301")
    nnunet_trained_model_3d_301 = safe_get("nnunet_trained_model_3d_301")
    nnunet_trained_model_2d_304 = safe_get("nnunet_trained_model_2d_304")
    rise_mask = safe_get("rise_mask")

    plots = safe_get("plots")

except Exception as e:
    print(f"An error occurred while loading the config: {e}")




def get_device() -> torch.device:
    """
    Determines the best available device (CUDA, DirectML or CPU).

    Returns:
        torch.device: The device to be used for computation.
    """
    if torch.cuda.is_available():
        logger.info("NVIDIA GPU found.")
        return torch.device("cuda")
    elif torch_directml.is_available():
        logger.info("AMD GPU found.")
        return torch_directml.device()
    else:
        logger.info("No GPU found, using CPU instead.")
        return torch.device("cpu")
