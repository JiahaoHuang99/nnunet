import os
import nibabel as nib
import numpy as np
import pandas as pd
from medpy.metric import binary

def list_nii_files(folder):
    """
    List all .nii or .nii.gz files in a folder
    """
    return sorted([f for f in os.listdir(folder) if f.endswith(".nii") or f.endswith(".nii.gz")])

def load_nifti(path):
    """
    Load NIfTI file and return binary mask
    """
    return nib.load(path).get_fdata().astype(np.uint8)

def compute_metrics_dsc(pred, gt, type='whole'):
    """
    Compute Dice Score for binary masks
    """
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    if type == 'whole':
        # label 1 & 2
        gt = ((gt == 1) | (gt == 2)).astype(np.uint8)
        pred = ((pred == 1) | (pred == 2)).astype(np.uint8)

    elif type == 'lesion':
        # label 2
        gt = (gt == 2).astype(np.uint8)
        pred = (pred == 2).astype(np.uint8)

    else:
        raise ValueError("Invalid type.")

    dsc = binary.dc(pred, gt)

    return dsc

def evaluate_matching_pairs(pred_folder, gt_folder, save_csv_path, save_csv_path_ave):
    """
    Evaluate all matching prediction-GT file pairs.
    """
    pred_files = list_nii_files(pred_folder)
    gt_files_set = set(list_nii_files(gt_folder))

    results = []
    missing_gt = []

    for pred_file in pred_files:
        if pred_file not in gt_files_set:
            missing_gt.append(pred_file)
            continue

        pred_path = os.path.join(pred_folder, pred_file)
        gt_path = os.path.join(gt_folder, pred_file)

        pred = load_nifti(pred_path)
        gt = load_nifti(gt_path)

        # WP: whole pancreas; PL: pancreas lesion
        dsc_wp = compute_metrics_dsc(pred, gt, type='whole')
        dsc_pl = compute_metrics_dsc(pred, gt, type='lesion')

        results.append({
            "case_name": pred_file,
            "DSC_WP": dsc_wp,
            "DSC_PL": dsc_pl
        })

    # Print Average
    if results:
        avg_dsc_wp = np.mean([r["DSC_WP"] for r in results])
        avg_dsc_pl = np.mean([r["DSC_PL"] for r in results])
        print(f"Average DSC WP: {avg_dsc_wp:.4f}, Average DSC PL: {avg_dsc_pl:.4f}")
    else:
        print("No valid predictions found.")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(save_csv_path, index=False)
    print(f"Saved {len(results)} results to {save_csv_path}")

    # Save average to separate CSV
    if results:
        df_ave = pd.DataFrame([{
            "DSC_WP": round(avg_dsc_wp, 4),
            "DSC_PL": round(avg_dsc_pl, 4)
        }])
        df_ave.to_csv(save_csv_path_ave, index=False)
        print(f"Saved average DSC results to {save_csv_path_ave}")

    if missing_gt:
        print(f"{len(missing_gt)} predictions have no matching GT and were skipped:")
        for f in missing_gt:
            print(f"  - {f}")

if __name__ == "__main__":


    pred_dir = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/validation"
    gt_dir = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw/ML-Quiz-3DMedImg/validation/labels"

    output_csv = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/result_seg.csv"
    output_csv_ave = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/result_seg_ave.csv"

    evaluate_matching_pairs(pred_dir, gt_dir, output_csv, output_csv_ave)
