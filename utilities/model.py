from utilities.patient_files import get_datasetname
import torch
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import config.config as cf
import numpy as np
from typing import Any, Dict

def load_model(dataset_id:int, config:str,fold:int=0, trainer:str="nnUNetTrainer", plan:str="nnUNetPlans", model_name: str="checkpoint_final.pth") -> Any:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=f"{cf.nnunet_trained_models}/{get_datasetname(dataset_id)}/{trainer}__{plan}__{config}",
        use_folds=(fold,),
    )
    model = predictor.network
    model = model.to(cf.get_device())
    model.eval()

    return model

def preprocess_input_for_inference(image: np.ndarray, patch_size=(256, 256)) -> torch.Tensor:

    image = (image - image.min()) / (image.max() - image.min())
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
    tensor_resized = F.interpolate(tensor, size=patch_size, mode='bilinear', align_corners=False)
    return tensor_resized

def pad_to_square(image_np, base=256, max_size=1024):
    h, w = image_np.shape

    target_size = base
    while target_size < max(h, w):
        target_size *= 2
        if target_size > max_size:
            raise ValueError(f"Bild ist zu groß für max_size={max_size}")

    if max(h, w) < base:
        target_size = base

    pad_h = target_size - h
    pad_w = target_size - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(
        image_np,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def unpad(padded, pads):
    pad_top, pad_bottom, pad_left, pad_right = pads
    return padded[pad_top:-pad_bottom or None, pad_left:-pad_right or None]


def pad_to_multiple(image, multiple=64):
    shape = np.array(image.shape[-3:])  # [Z, Y, X]
    target_shape = ((shape + multiple - 1) // multiple) * multiple
    pad_before = (target_shape - shape) // 2
    pad_after = target_shape - shape - pad_before

    pad_width = [
        (0, 0),
        (pad_before[0], pad_after[0]),
        (pad_before[1], pad_after[1]),
        (pad_before[2], pad_after[2]),
    ]

    padded = np.pad(image, pad_width, mode='constant')
    return padded, pad_width


