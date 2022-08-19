##############################################################################################
python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_it -o /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_it/ -r True -in True -in_d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/
#epoch<=3 : loss = loss_1
#epoch>3: loss = loss_1 + 0.0001 * loss_2
'CUDA' : 1
# 并且调整了pnp-SOLVER的参数
#############################################################################################
python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5/epoch_10.pth -d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/ -o /mnt/data/Dream_ty/Dream_model/infer_test -b 16 -w 8


#############################################################################################
python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_ls2 -o /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_ls2/ -in True -in_d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/

#loss = loss_2
'CUDA':2



