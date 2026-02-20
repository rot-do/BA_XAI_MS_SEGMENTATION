import os
from pathlib import Path

import nibabel as nib
from nibabel.filebasedimages import FileBasedImage
import csv
import config.config as cf
import re

def get_patient(dataset_id:int, patient_id: int) -> tuple[FileBasedImage, FileBasedImage]:
    img_filename = f"Patient-{patient_id:02d}_0000.nii.gz"
    label_filename = f"Patient-{patient_id:02d}.nii.gz"

    img_path_tr = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "imagesTr", img_filename)
    img_path_ts = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "imagesTs", img_filename)
    label_path_tr = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "labelsTr", label_filename)
    label_path_ts = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "labelsTs", label_filename)

    if os.path.exists(img_path_tr) and os.path.exists(label_path_tr):
        flair_nib = nib.load(img_path_tr)
        mask_nib = nib.load(label_path_tr)
        cf.logger.info(f"Loaded training data for Patient-{patient_id:02d}")
    elif os.path.exists(img_path_ts) and os.path.exists(label_path_ts):
        flair_nib = nib.load(img_path_ts)
        mask_nib = nib.load(label_path_ts)
        cf.logger.info(f"Loaded test data for Patient-{patient_id:02d}")
    else:
        img_filename = f"Patient-{patient_id}_0000.nii.gz"
        label_filename = f"Patient-{patient_id}.nii.gz"

        img_path_tr = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "imagesTr", img_filename)
        img_path_ts = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "imagesTs", img_filename)
        label_path_tr = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "labelsTr", label_filename)
        label_path_ts = os.path.join(cf.nnUNet_raw, get_datasetname(dataset_id), "labelsTs", label_filename)

        if os.path.exists(img_path_tr) and os.path.exists(label_path_tr):
            flair_nib = nib.load(img_path_tr)
            mask_nib = nib.load(label_path_tr)
            cf.logger.info(f"Loaded training data for Patient-{patient_id:02d}")
        elif os.path.exists(img_path_ts) and os.path.exists(label_path_ts):
            flair_nib = nib.load(img_path_ts)
            mask_nib = nib.load(label_path_ts)
            cf.logger.info(f"Loaded test data for Patient-{patient_id:02d}")
        else:
            error_msg = f"Patient-{patient_id:02d} not found in training or test sets."
            cf.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    return flair_nib, mask_nib

def get_datasetname(dataset_id: int) -> str:
    attr_name = f"dataset_{dataset_id}"
    if hasattr(cf, attr_name):
        return getattr(cf, attr_name)
    else:
        raise Exception(f"Dataset with ID {dataset_id} not found")


_csv_path = Path(f"{cf.nnUNet_raw}/{cf.dataset_200}/renaming_history.csv")
_mapping = None
def _load_mapping():
    global _mapping
    if _mapping is not None:
        return
    if not os.path.exists(_csv_path):
        raise FileNotFoundError(f"CSV-Datei nicht gefunden: {_csv_path}")

    _mapping = {}
    with open(_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2:
                continue
            original, new = row
            original = original.strip()
            new = new.strip()

            original_match = re.search(r'Patient-(\d+)', original)
            new_match = re.search(r'Patient-(\d+)', new)
            center_match = re.search(r'center_(\d+)', original)

            if not (original_match and new_match and center_match):
                continue

            original_num = int(original_match.group(1))
            new_num = int(new_match.group(1))
            center_num = int(center_match.group(1))

            _mapping[new_num] = (original_num, center_num)


def get_center_patient(new_id):
    _load_mapping()

    if isinstance(new_id, str) and new_id.lower().startswith("patient-"):
        num_match = re.search(r'Patient-(\d+)', new_id)
        if num_match:
            new_id = int(num_match.group(1))
        else:
            return None

    return _mapping.get(new_id, None)