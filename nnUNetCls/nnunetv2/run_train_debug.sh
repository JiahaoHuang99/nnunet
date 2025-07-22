export nnUNet_raw="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw"
export nnUNet_preprocessed="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_preprocessed"
export nnUNet_results="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results"

rm log_train_100_3d_fullres_cls_lambda0p5_0.txt
CUDA_VISIBLE_DEVICES=0 \
nohup nnUNetv2_train 100 3d_fullres 0 --npz -tr nnUNetClsTrainer_Lambda0p5 -p nnUNetResEncUNetMCLSPlans >> log_train_100_3d_fullres_cls_lambda0p5_0.txt &
