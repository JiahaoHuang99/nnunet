import pandas as pd
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def extract_gt_label(case_name: str) -> int:
    """
    Extract ground truth label from the case_name using regex.
    For example, 'quiz_0_168' -> 0
    """
    match = re.match(r"quiz_(\d)_\d+", case_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid case_name format: {case_name}")

def collect_results(df: pd.DataFrame):
    """
    Extract GT and prediction lists from the DataFrame.
    """
    df['gt'] = df['case_name'].apply(extract_gt_label)
    y_true = df['gt'].tolist()
    y_pred = df['pred'].tolist()
    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy and macro F1, return a dict.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-Averaged F1 Score: {macro_f1:.4f}")

    result_dict = {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

    return result_dict

def save_metrics_to_csv(metrics_dict: dict, output_csv_path: str):
    """
    Save the metrics dictionary to a single-row CSV file.
    """
    df_metrics = pd.DataFrame([metrics_dict])
    df_metrics.to_csv(output_csv_path, index=False)
    print(f"\nMetrics saved to: {output_csv_path}")


if __name__ == "__main__":

    src_csv_path = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/result_cls.csv"
    csv_path = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/result_cls_ave.csv"

    df = pd.read_csv(src_csv_path)
    y_true, y_pred = collect_results(df)

    result_dict = compute_metrics(y_true, y_pred)

    save_metrics_to_csv(result_dict, csv_path)
