# README

A note for Jiahao Huang


1. Setup Env

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md


```
conda create -n nnunetv2_cls python==3.11.5
cd .../nnunet/nnUNetCLS
pip install -e .
cd  .../nnunet/dynamic-network-architectures
pip install -e .
```


2. Setup Path

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md

`vim ~/.bashrc`

add:
```sh
export nnUNet_raw="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw"
export nnUNet_preprocessed="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_preprocessed"
export nnUNet_results="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results"
```


3. Prepare Dataset

3.1 Generate Data for nnUNet_raw

Run:
`.../nnunet/nnUNetCLS/dataset_conversion/Dataset100_JM.py`

3.2 Preprocessing

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#experiment-planning-and-preprocessing
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md

Run:
```sh
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
```

```sh
nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)
```

4. Train

See:

```sh
.../nnunet/nnUNetCLS/run_train_debug.sh
```


5. Val

See:
```sh
.../nnunet/nnUNetCLS/run_test_debug.sh
```

6. Cal metrics

See:

```sh
.../nnunet/nnUNetCLS/cal_metrics_cls.py
.../nnunet/nnUNetCLS/cal_metrics_seg.py
```
