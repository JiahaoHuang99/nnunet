export nnUNet_raw="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw"
export nnUNet_preprocessed="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_preprocessed"
export nnUNet_results="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results"

rm log_train_100_3d_fullres_0.txt
CUDA_VISIBLE_DEVICES=0 \
nohup nnUNetv2_train 100 3d_fullres 0 --npz -tr nnUNetTrainer -p nnUNetResEncUNetMPlans >> log_train_086_3d_fullres_0.txt &

rm log_train_100_3d_fullres_cls_0.txt
CUDA_VISIBLE_DEVICES=1 \
nohup nnUNetv2_train 100 3d_fullres 0 --npz -tr nnUNetClsTrainer -p nnUNetResEncUNetMCLSPlans >> log_train_086_3d_fullres_cls_0.txt &


#nnUNetv2_train 100 3d_fullres 0 --npz -tr nnUNetClsTrainerDEBUG -p nnUNetResEncUNetMCLSPlans
