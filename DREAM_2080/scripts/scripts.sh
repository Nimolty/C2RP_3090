#############################################################################################
#python scripts/train_network.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q_key.yaml -e 25 -lr 0.00015 -b 32 -w 16 -o /Huge_Disk/Dream_data_0809/ckpt_model/ -n 7toEpro2

#python scripts/train_network.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q_key.yaml -e 25 -lr 0.00015 -b 32 -w 16 -n 7toEpro2
#
#python scripts/train_network.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q_key.yaml -e 25 -lr 0.00015 -b 32 -w 16 -n 7toEpro2
#
#python scripts/train_network.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q_key.yaml -e 25 -lr 0.00015 -b 32 -w 16 -n 7toEpro2

#python scripts/train_network.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -n 7toEpro0812 -o /Huge_Disk/Dream_data_0809/ckpt_model/

# 此时loss_t和loss_r之前的系数都是0.1

##############################################################################################
#python scripts/train_network.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_train_bs32_lr00001_s0 -o /Huge_Disk/Dream_data_0809/ckpt_syn_train_bs32_lr00001_s0
#
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_train_bs32_lr00001_s0 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr00001_s0

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_test_bs32_lr00001_s0 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s0 -r True
#
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_test_bs32_lr00001_s0 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s0 -r True -rt 3 -ad_lr 5e-5

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_test_bs32_lr00001_s0 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s0

#python scripts/test_adlr.py -i /Huge_Disk/Dream_data_0809/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 0 -n syn_train_bs32_lr00001_s0_test -o /Huge_Disk/Dream_data_0809/ckpt_syn_train_bs32_lr00001_s0_test -r True -rt 4 -ad_lr 1e-5

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr00001_s5

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 16 -w 16 -s 5 -n syn_test_bs16_lr000015_s5 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs16_lr000015_s5 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
## CUDA:3
#
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 16 -w 16 -s 5 -n graph_test 
# CUDA:1

#python scripts/network_inference_dataset.py -i /mnt/data/Dream_ty/Dream_ckpt/ckpt_syn_test_bs32_lr00001_s5/best_network.pth -d /mnt/data/Dream_data_all/data/real/panda-3cam_realsense/ -o /mnt/data/Dream_ty/infer_dataset_test -b 16 -w 8

#python scripts/network_inference_dataset.py -i /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s5_wols2/best_network.pth -d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/ -o /Huge_Disk/Dream_data_0809/Dream_model/infer_test -b 16 -w 8

python scripts/network_inference_dataset.py -i /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5/epoch_9.pth -d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/ -o /Huge_Disk/Dream_data_0809/Dream_model/infer_test_cross_e9 -b 16 -w 8

python scripts/network_inference_dataset.py -i /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_ls1/epoch_7.pth -d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/ -o /Huge_Disk/Dream_data_0809/Dream_model/infer_test_cross_e7_ls1 -b 16 -w 8
#
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_test_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n infer_test -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/infer_test -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5_new -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr00001_s5_new -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/

#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n cuda_test -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/cuda_test -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 26 -w 16 -s 5 -n cuda_test_1 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/cuda_test_1
#loss = loss_1 + (0.02 * loss_mc + 0.1 * loss_t + 0.1 *loss_r)
##############################################################################################
##############################################################################################
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5_ls1 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s5_ls1 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/

#loss = loss_1
##############################################################################################
##############################################################################################
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_test_bs32_lr00001_s5 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/

#self.head0加上softmax
#epoch=1 : loss = 10 * loss_1
#epoch>=2: loss = 10 * loss_1 + loss_2
'CUDA' : 3
##############################################################################################
##############################################################################################
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5_ls1_0817 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr00001_s5_ls1_0817 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/

#self.head0加上softmax 16x7x100x100 nn.Softmax(dim=-1)
#loss = 10 * loss_1
'CUDA' : 2
##############################################################################################
#python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.0001 -b 32 -w 16 -s 5 -n syn_train_bs32_lr00001_s5_div100 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr00001_s5_div100 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#self.head0加上softmax
#loss = loss_1 + loss_2
# x3d除以了100
'CUDA' : 1
##############################################################################################
python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#epoch<=3 : loss = loss_1
#epoch>3: loss = loss_1 + 0.0001 * loss_2
'CUDA' : 3


##############################################################################################
python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_ls1 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_ls1 -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#loss = loss_1
'CUDA' : 2

##############################################################################################
#output_head = output_head.view(bs, -1, 2, H, W)
#softmax = nn.Softmax(dim=2)
#output_head = softmax(output_head)
#output_head_0 = output_head[:, :, 0, :, :]
#做了上述softmax的结果
python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_sfmls1 -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_sfmls1 -r True -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#loss = loss_1
'CUDA' : 1
##############################################################################################
#output_head = output_head.view(bs, -1, 2, H, W)
#softmax = nn.Softmax(dim=2)
#output_head = softmax(output_head)
#output_head_0 = output_head[:, :, 0, :, :]
#做了上述softmax的结果
python scripts/train_network_r.py -i /Huge_Disk/Dream_data_all/data/synthetic/panda_synth_train_dr/ -t 0.8 -m manip_configs/panda.yaml -ar arch_configs/dream_vgg_q.yaml -e 30 -lr 0.00015 -b 32 -w 16 -s 5 -n syn_train_bs32_lr000015_s5_sfm -o /Huge_Disk/Dream_data_0809/Dream_model/ckpt/ckpt_syn_train_bs32_lr000015_s5_sfm -r True -in True -in_d /Huge_Disk/Dream_data_all/data/real/panda-3cam_realsense/
#epoch<=3 : loss = loss_1
#epoch>3: loss = loss_1 + 0.0001 * loss_2
'CUDA' : 0


##############################################################################################

