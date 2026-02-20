import os
import nibabel as nib
import config.config as cf

def convert_nii_to_niigz(input_folder: str, delete_original: bool = False) -> None:
    nii_files = [f for f in os.listdir(input_folder) if f.endswith(".nii") and not f.endswith(".nii.gz")]

    if not nii_files:
        cf.logger.info("No .nii files found.")
        return

    for nii_file in nii_files:
        nii_path = os.path.join(input_folder, nii_file)
        gz_path = nii_path + ".gz"
        cf.logger.info(f"Converting: {nii_file} → {os.path.basename(gz_path)}")

        img = nib.load(nii_path)
        nib.save(img, gz_path)

        if delete_original:
            os.remove(nii_path)
            cf.logger.info(f"Original file deleted: {nii_file}")

    cf.logger.info("Conversion completed.")


if __name__ == "__main__":
    folder_path = r""
    convert_nii_to_niigz(folder_path, delete_original=False)
