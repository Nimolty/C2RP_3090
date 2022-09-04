# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:52:39 2022

@author: lenovo
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tools._init_paths as _init_paths

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import cv2
import json
import copy
import numpy as np
from lib.opts import opts
from lib.Dream_detector import DreamDetector
import torch
from tqdm import tqdm
import dream_geo as dream

keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]

def find_dataset(opt):
    input_dir = opt.dataset
    input_dir = os.path.expanduser(input_dir) # 输入的是../../franka_data_0825
    assert os.path.exists(input_dir),\
    'Expected path "{}" to exist, but it does not.'.format(input_dir)
    dirlist = os.listdir(input_dir) # 现在变成List 从00000到02000了，目前生成了2000个视频序列了
    
    found_videos = []
    for each_dir in dirlist:
        output_dir = os.path.join(input_dir, each_dir)
        # output_dir = ../../franka_data_0825/xxxxx
        found_video = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                       if f.endswith('.png')]
        found_json = [os.path.join(output_dir, f) for f in os.listdir(output_dir) \
                        if f.endswith("meta.json")]
        found_video.sort()
        found_json.sort()
        if len(found_video) != 30 or len(found_json) != 30:
            continue
        found_videos.append([found_video, found_json])
    
    return found_videos

def inference(opt):
    found_videos = find_dataset(opt)
    json_list, detected_kps_list = [], []
    for found_video_0 in tqdm(found_videos[200:300]):
        # found_video_0 = found_videos[j]
        # print('found_video_0', found_video_0) 
        # print('json_path', found_video_0[1])
        with torch.no_grad():
            detector = DreamDetector(opt, found_video_0[1][-1], keypoint_names)
            length = len(found_video_0[0])
            # print(length)
            for i, img_path in enumerate(found_video_0[0]):
                json_path = found_video_0[1][i]
                img = cv2.imread(img_path)
                if i != length - 1:
                    ret = detector.run(img, i, json_path)
                else:
                    ret, detected_kps = detector.run(img, i, json_path, is_final=True)
                    detected_kps_np = np.array(detected_kps)
                    output_dir = img_path.rstrip('png')
                    np.savetxt(output_dir + 'txt', detected_kps_np)
                    json_list.append(json_path)
                    detected_kps_list.append(detected_kps_np)
                    # print(detected_kps)
    
    exp_dir = opt.exp_dir
    pth_order = opt.load_model.split('/')[-1]
    exp_id = opt.load_model.split('/')[-3]
    pth_order = pth_order.rstrip('.pth')
    output_dir = os.path.join(exp_dir, exp_id,'results', pth_order, '')
    dream.utilities.exists_or_mkdir(output_dir)
    
    dream.analysis.analyze_ndds_center_dream_dataset(
    json_list, # 在外面直接写一个dataset就好了，需要注意它的debug_node为LIGHT
    detected_kps_list,
    opt, 
    keypoint_names,
    [640, 480],
    output_dir)
    
                # print('ret', ret)
 
 

if __name__ == "__main__":
    opt = opts().init_infer(7, (480, 480))
    inference(opt)
    
























