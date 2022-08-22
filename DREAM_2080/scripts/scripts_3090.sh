##############################################################################################
#python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_21 -o /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_21/
#epoch<=3 : loss = loss_1
#epoch>3: loss = loss_1 + 0.0001 * loss_2
'CUDA' : 1
# 并且调整了pnp-SOLVER的参数
############################################################################################
##小批量数据，看一下效果，开始调参(取消validation环节)
#epoch<=3 : loss = loss_1
#epoch>3: loss = loss_1 + 0.0001 * loss_2
#python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test/
#
#python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_prev -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_prev/
#
python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw/

#############################################################################################

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw_new -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw_new/
# new w2d

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw_heat -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw_heat/
#if this_epoch <= 3:
#    loss = loss_1 
#else:
#    loss = loss_1 +  0.0001 * loss_2 + 10 * loss_w

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw_1 -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw_1/

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_img -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_img/
#if this_epoch <= 3:
#    loss = loss_1 
#else:
#    loss = loss_1 +  0.0001 * loss_2 + 1 * loss_w
##############################################################################################
python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw/

#if this_epoch <= 3:
#    loss = loss_1 
#else:
#    loss = loss_1 +  0.0001 * loss_2 + 100 * loss_w
# new:w2d

##############################################################################################

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_lsw_100 -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_lsw_100/
#if this_epoch <= 3:
#    loss = loss_1 
#else:
#    loss = loss_1 +  0.0001 * loss_2 + 100 * loss_w

##############################################################################################

python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.025 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 50 -lr 0.00015 -b 32 -w 16 -s 5 -n param_test_wolsw -o /mnt/data/Dream_ty/Dream_model/ckpt/param_test_wolsw/
#if this_epoch <= 3:
#    loss = loss_1 
#else:
#    loss = loss_1 +  0.0001 * loss_2 

#############################################################################################
#python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_21/epoch_1.pth -d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/ -o /mnt/data/Dream_ty/Dream_model/infer_test -b 16 -w 8

python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_it/epoch_24.pth -d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/ -o /mnt/data/Dream_ty/Dream_model/infer_test -b 16 -w 8

python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_it/epoch_24.pth -d /mnt/data/Dream_data_all/data/synthetic/panda_synth_test_dr -o /mnt/data/Dream_ty/Dream_model/infer_test_syn -b 16 -w 8

python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_it/epoch_24.pth -d /mnt/data/Dream_data_all/data/synthetic/panda_synth_test_dr -o /mnt/data/Dream_ty/Dream_model/infer_test_syn_short -b 16 -w 8
#############################################################################################
#python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_ls2 -o /mnt/data/Dream_ty/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_ls2/ -in True -in_d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/

#loss = loss_2
'CUDA':2

#python scripts/train_network_r.py -i /mnt/data/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n test -o /mnt/data/Dream_ty/Dream_model/ckpt/test/ -in True -in_d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/