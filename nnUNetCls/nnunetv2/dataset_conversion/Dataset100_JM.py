import os
import shutil

def distribute_nii_files(path_a, path_b, path_c):
    # 创建目标文件夹（如果不存在）
    os.makedirs(path_b, exist_ok=True)
    os.makedirs(path_c, exist_ok=True)

    # 遍历 path_a 中所有 .nii.gz 文件
    for filename in os.listdir(path_a):
        if filename.endswith(".nii.gz"):
            src = os.path.join(path_a, filename)
            if "_0000" in filename:
                dst = os.path.join(path_b, filename)
            else:
                dst = os.path.join(path_c, filename)
            shutil.copy2(src, dst)
            print(f"Copied {filename} to {dst}")

# 用法示例
if __name__ == "__main__":
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype0"
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype1"
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype2"
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype0"
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype1"
    # path_a = "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype2"

    for path_a in [
        # "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype0",
        "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype1",
        "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/train/subtype2",
        "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype0",
        "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype1",
        "/media/NAS07/USER_PATH/jh/ML-Quiz-3DMedImg/validation/subtype2",
    ]:

        path_b = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw/Dataset100_JM/imagesTr"
        path_c = "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw/Dataset100_JM/labelsTr"
        distribute_nii_files(path_a, path_b, path_c)
