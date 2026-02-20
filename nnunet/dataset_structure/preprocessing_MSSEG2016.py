import os
import shutil
import nibabel as nib
import csv
from typing import List
import config.config as cf
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def create_directories(raw_data_path: str, dataset: str) -> None:
    cf.logger.info(f"Creating subdirectories for {dataset}")
    subdirs = ["preparation/mri/train", "preparation/mask/train", "preparation/mri/test", "preparation/mask/test",
               "imagesTr", "labelsTr", "imagesTs", "labelsTs"]
    for subdir in subdirs:
        full_path = os.path.join(raw_data_path, subdir)
        if os.path.exists(full_path):
            cf.logger.info(f"Deleting existing subdirectory: {subdir}")
            for f in os.listdir(full_path):
                f_path = os.path.join(full_path, f)
                if os.path.isfile(f_path) or os.path.islink(f_path):
                    os.unlink(f_path)
                elif os.path.isdir(f_path):
                    shutil.rmtree(f_path)
        os.makedirs(full_path, exist_ok=True)
    cf.logger.info(f"Subdirectories for {dataset} created")


def preparing_msseg2016(dataset: str, log_path: str, start_id: int = 0) -> int:
    cf.logger.info("Collecting patient files of MSSEG2016")

    patients_dict = {}
    centers_train = ["Center_01", "Center_07", "Center_08"]
    for center in centers_train:
        path = os.path.join(cf.msseg2016_train, center)
        patients_dict[f"{center.lower()}_train"] = sorted([
            os.path.join(path, p) for p in os.listdir(path)
            if os.path.isdir(os.path.join(path, p))
        ])
    centers_test = ["Center_01", "Center_03", "Center_07", "Center_08"]

    for center in centers_test:
        path = os.path.join(cf.msseg2016_test, center)
        patients_dict[f"{center.lower()}_test"] = sorted([
            os.path.join(path, p) for p in os.listdir(path)
            if os.path.isdir(os.path.join(path, p))
        ])

    for center_name, patients in patients_dict.items():
        start_id = _process_center(patients, center_name, dataset, log_path, start_id)

    cf.logger.info("Patient files of MSSEG2016 dataset copied")
    return start_id


def _process_center(patients: List[str], center_name: str, dataset: str, log_path: str, start_id: int) -> int:
    for patient in patients:
        start_id += 1
        patient_id = int(patient.split('_')[-1])
        cf.logger.info(f"Copying patient-{patient_id} of {center_name}")

        if "200" in dataset:
            flair_file = os.path.join(patient, 'Raw_Data', 'bias_corrected', 'FLAIR_preprocessed_withskull.nii.gz')
            dp_file = os.path.join(patient, 'Raw_Data', 'bias_corrected', 'DP_preprocessed_withskull.nii.gz')
            gado_file = os.path.join(patient, 'Raw_Data', 'bias_corrected', 'GADO_preprocessed_withskull.nii.gz')
            t1_file = os.path.join(patient, 'Raw_Data', 'bias_corrected', 'T1_preprocessed_withskull.nii.gz')
            t2_file = os.path.join(patient, 'Raw_Data', 'bias_corrected', 'T2_preprocessed_withskull.nii.gz')
            mask_file = os.path.join(patient, 'Masks', 'registered', 'Raw_Consensus.nii.gz')
        elif "201" in dataset:
            flair_file = os.path.join(patient, 'Preprocessed_Data', 'reoriented',
                                      'FLAIR_preprocessed_reoriented.nii.gz')
            dp_file = os.path.join(patient, 'Preprocessed_Data', 'reoriented', 'DP_preprocessed_reoriented.nii.gz')
            gado_file = os.path.join(patient, 'Preprocessed_Data', 'reoriented', 'GADO_preprocessed_reoriented.nii.gz')
            t1_file = os.path.join(patient, 'Preprocessed_Data', 'reoriented', 'T1_preprocessed_reoriented.nii.gz')
            t2_file = os.path.join(patient, 'Preprocessed_Data', 'reoriented', 'T2_preprocessed_reoriented.nii.gz')
            mask_file = os.path.join(patient, 'Masks', 'reoriented', 'Consensus_reoriented.nii.gz')
        else:
            raise Exception(f"wrong dataset! {dataset}")

        flair_nii = nib.load(flair_file)
        dp_nii = nib.load(dp_file)
        gado_nii = nib.load(gado_file)
        t1_nii = nib.load(t1_file)
        t2_nii = nib.load(t2_file)
        mask_nii = nib.load(mask_file)

        patient_str = f"Patient-{start_id:02d}"
        new_dp_name = f"{patient_str}_0000.nii.gz"
        new_gado_name = f"{patient_str}_0001.nii.gz"
        new_t1_name = f"{patient_str}_0002.nii.gz"
        new_t2_name = f"{patient_str}_0003.nii.gz"
        new_flair_name = f"{patient_str}_0004.nii.gz"
        new_mask_name = f"{patient_str}.nii.gz"

        suffix = center_name.split("_")[-1]

        new_dp_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mri/{suffix}", new_dp_name)
        new_gado_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mri/{suffix}", new_gado_name)
        new_t1_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mri/{suffix}", new_t1_name)
        new_t2_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mri/{suffix}", new_t2_name)
        new_flair_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mri/{suffix}", new_flair_name)
        new_mask_path = os.path.join(cf.nnUNet_raw, dataset, "preparation", f"mask/{suffix}", new_mask_name)

        nib.save(dp_nii, new_dp_path)
        nib.save(gado_nii, new_gado_path)
        nib.save(t1_nii, new_t1_path)
        nib.save(t2_nii, new_t2_path)
        nib.save(flair_nii, new_flair_path)
        nib.save(mask_nii, new_mask_path)

        log_renaming_history(
           log_path,
           f"{center_name}/Patient-{patient_id}",
           f"Patient-{start_id:02d}",
        )

    return start_id


def gather_patients(dataset: str, split: str) -> List[str]:
    mri_dir = os.path.join(cf.nnUNet_raw, dataset, "preparation", "mask", split)
    patients = sorted([
        os.path.join(mri_dir, p) for p in os.listdir(mri_dir)
        if p.endswith(".nii.gz")
    ])

    return patients


def copy_nnunet(patients: List[str], target_folder: str, dataset: str, with_labels: bool = True) -> None:

    for patient in patients:
        patient_filename = os.path.basename(patient).replace(".nii.gz", "")
        parent_dir = os.path.basename(os.path.dirname(patient))
        patient_id = patient_filename.split("-")[1]
        cf.logger.info(f"Copying patient {patient_id}")
        cf.logger.info(
            f"Copying patient {patient_id} to {cf.nnUNet_raw}/{dataset}/preparation/mri/{patient_filename}_0000.nii.gz")

        dp_path = f"{cf.nnUNet_raw}/{dataset}/preparation/mri/{parent_dir}/{patient_filename}_0000.nii.gz"
        gado_path = f"{cf.nnUNet_raw}/{dataset}/preparation/mri/{parent_dir}/{patient_filename}_0001.nii.gz"
        t1_path = f"{cf.nnUNet_raw}/{dataset}/preparation/mri/{parent_dir}/{patient_filename}_0002.nii.gz"
        t2_path = f"{cf.nnUNet_raw}/{dataset}/preparation/mri/{parent_dir}/{patient_filename}_0003.nii.gz"
        flair_path = f"{cf.nnUNet_raw}/{dataset}/preparation/mri/{parent_dir}/{patient_filename}_0004.nii.gz"

        flair_nii = nib.load(flair_path)
        dp_nii = nib.load(dp_path)
        gado_nii = nib.load(gado_path)
        t1_nii = nib.load(t1_path)
        t2_nii = nib.load(t2_path)

        flair_target_path = os.path.join(cf.nnUNet_raw, dataset, f"images{target_folder}", os.path.basename(flair_path))
        dp_target_path = os.path.join(cf.nnUNet_raw, dataset, f"images{target_folder}", os.path.basename(dp_path))
        gado_target_path = os.path.join(cf.nnUNet_raw, dataset, f"images{target_folder}", os.path.basename(gado_path))
        t1_target_path = os.path.join(cf.nnUNet_raw, dataset, f"images{target_folder}", os.path.basename(t1_path))
        t2_target_path = os.path.join(cf.nnUNet_raw, dataset, f"images{target_folder}", os.path.basename(t2_path))

        nib.save(flair_nii, flair_target_path)
        nib.save(dp_nii, dp_target_path)
        nib.save(gado_nii, gado_target_path)
        nib.save(t1_nii, t1_target_path)
        nib.save(t2_nii, t2_target_path)

        if with_labels:
            mask_file = patient.replace(os.path.join("mri"), os.path.join("mask")).replace("_0000", "")
            mask_filename = os.path.basename(mask_file)
            mask_nii = nib.load(mask_file)
            mask_target_path = os.path.join(cf.nnUNet_raw, dataset, f"labels{target_folder}", mask_filename)
            nib.save(mask_nii, mask_target_path)


def create_dataset_json(num_train_cases: int, raw_data_path: str, dataset: str) -> None:

    generate_dataset_json(
        output_folder=raw_data_path,
        channel_names={0: "DP",
                       1: "GADO",
                       2: "T1",
                       3: "T2",
                       4: "FLAIR"},
        labels={"background": 0, "lesion": 1},
        num_training_cases=num_train_cases,
        file_ending=".nii.gz",
        dataset_name=dataset,
        description="Multiple Sclerosis lesion segmentation based on FLAIR, T1, T2, DP and GADO with given train/test split",
        reference="Bachelor thesis dataset",
        converted_by="Dominik"
    )


def log_renaming_history(
        csv_path: str,
        old_id: str,
        new_id: str,
        header: bool = False
) -> None:

    mode = 'w' if header else 'a'
    with open(csv_path, mode=mode, newline='') as csvfile:
        writer = csv.writer(csvfile)
        if header:
            writer.writerow(["Original Patient ID", "New Patient ID"])
        else:
            writer.writerow([old_id, new_id])


def create_dataset(dataset: str, log_path: str) -> None:
    raw_data_path = os.path.join(cf.nnUNet_raw, dataset)

    create_directories(raw_data_path, dataset)
    log_renaming_history(log_path, "", "", header=True)

    preparing_msseg2016(dataset, log_path)

    train = gather_patients(dataset, "train")
    test = gather_patients(dataset, "test")

    copy_nnunet(train, "Tr", dataset, with_labels=True)
    copy_nnunet(test, "Ts", dataset, with_labels=True)

    create_dataset_json(len(train), raw_data_path, dataset)


if __name__ == "__main__":
    dataset = cf.dataset_201
    log_path = os.path.join(cf.nnUNet_raw, dataset, "renaming_history.csv")
    create_dataset(dataset, log_path)

    dataset = cf.dataset_200
    log_path = os.path.join(cf.nnUNet_raw, dataset, "renaming_history.csv")
    create_dataset(dataset, log_path)
