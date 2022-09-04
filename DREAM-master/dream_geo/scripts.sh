###########################################################################################
# python Dream_main.py tracking --exp_id 1  --pre_hm --same_aug --resume --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1,3 --model_last_pth model_40.pth
#
#python Dream_main.py tracking --exp_id 1  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 1,3
#
#python Dream_main.py tracking --exp_id 2  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3

#python Dream_main.py tracking --exp_id 3  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_9.pth
#
#python Dream_main.py tracking --exp_id 4  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.1 --fp_disturb 0.1 --gpus 3 --resume --model_last_pth model_14.pth
# 这里tracking loss乘以了0.01

python Dream_ct_inference.py tracking --load_model /mnt/data/Dream_ty/Dream_model/center-dream/tracking/4/ckpt/model_12.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 3

python Dream_ct_inference.py tracking --load_model /mnt/data/Dream_ty/Dream_model/center-dream/tracking/4/ckpt/model_6.pth --pre_hm --track_thresh 0.001 --test_focal_length 633 --gpus 0


python Dream_main.py tracking --exp_id 4  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3

python Dream_main.py tracking --exp_id 5  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 2


###########################################################################################
python Dream_main.py tracking --exp_id 6  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 2
# tracking loss * 0.01, reg loss * 0.1
###########################################################################################
python Dream_main.py tracking --exp_id 7  --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 3
# tracking loss * 0.01, reg loss * 0.01


###########################################################################################