# README

A note for Jiahao Huang


1. Setup Env

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md


```
conda create -n nnunetv2_cls python==3.11.5
cd /mnt/workspace/aneurysm/nnUNetCLS
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
`.../dataset_conversion/Dataset085_artery_cta_aneu_base.py`
`.../dataset_conversion/Dataset086_artery_nct_aneu_base.py`

3.2 Preprocessing

https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#experiment-planning-and-preprocessing
https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md

Run:
```sh
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
```

```sh
nnUNetv2_plan_experiment -d DATASET -pl nnUNetPlannerResEnc(M/L/XL)

nnUNetv2_plan_experiment -d 100 -pl nnUNetPlannerResEncM
```

4. Train

Run:

```sh
cd aneurysm/nnUNet/nnunetv2
```

```sh

nohup nnUNetv2_train 085 3d_fullres 0 --npz -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans >> log_train_085_3d_fullres_0.txt & 

nohup nnUNetv2_train 086 3d_fullres 0 --npz -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans >> log_train_086_3d_fullres_0.txt & 
```




5. Val


Run:

```sh
cd aneurysm/nnUNet/nnunetv2
```

```sh
nohup nnUNetv2_train 085 3d_fullres 0 --npz -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans --val --val_best >> log_test_085_3d_fullres_0.txt & 

nohup nnUNetv2_train 086 3d_fullres 0 --npz -tr nnUNetTrainer -p nnUNetResEncUNetXLPlans --val --val_best >> log_test_086_3d_fullres_0.txt & 
```



6. Evaluation

```bash
/mnt/workspace/aneurysm/nnUNet/evaluation/batch_metric_new_npz_allfold.py
```