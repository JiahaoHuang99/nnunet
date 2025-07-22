export nnUNet_raw="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw"
export nnUNet_preprocessed="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_preprocessed"
export nnUNet_results="/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results"

python -m inference.predict_from_raw_data_cls \
-i "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_raw/ML-Quiz-3DMedImg/test" \
-o "/media/NAS07/USER_PATH/jh/nnunet/test_junma/nnUNetCls_output/nnUNet_results/Dataset100_JM/nnUNetClsTrainer_Lambda0p5__nnUNetResEncUNetMCLSPlans__3d_fullres/fold_0/test" \
-d 100 -p nnUNetResEncUNetMCLSPlans -c 3d_fullres -f 0 -chk "checkpoint_best.pth" -tr nnUNetClsTrainer_Lambda0p5 -step_size 0.5 --disable_tta

