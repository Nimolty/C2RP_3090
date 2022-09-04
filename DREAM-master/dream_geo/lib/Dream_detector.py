# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 00:33:32 2022

@author: lenovo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import numpy as np
from progress.bar import Bar
import time
import torch
import math
import os
from ruamel.yaml import YAML
from copy import deepcopy
from PIL import Image

from model.model import create_model, load_model
from model.decode import dream_generic_decode
from model.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform, affine_transform
from utils.image import dream_draw_umich_gaussian, gaussian_radius
from utils.post_process import dream_generic_post_process
from utils.debugger import Debugger
from utils.tracker import Tracker
from dataset.dataset_factory import get_dataset
import dream_geo as dream

class DreamDetector(object):
    def __init__(self, opt, output_dir, keypoint_names):
        if opt.gpus[0] >= 0:
            opt.device = torch.device("cuda")
        else:
            opt.device = torch.device("cpu")
        
        print("Creating model ...")
        # print(f'opt.arch : {opt.arch}, opt.heads : {opt.heads}, opt.head_conv : {opt.head_conv}')
        self.model = create_model(
            opt.arch, opt.heads, opt.head_conv, opt=opt)
        self.model = load_model(self.model, opt.load_model, opt)
        self.model = self.model.to(opt.device)
        self.model.eval()
        
        self.opt = opt
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(1, 1, 3)
        self.pause = not opt.no_pause
        self.rest_focal_length =  self.opt.test_focal_length
        # self.flip_idx = self.trained_dataset.flip_idx
        self.cnt = 0
        self.pre_images = None
        self.pre_image_ori = None
        self.tracker = Tracker(opt)
        self.output_dir = output_dir.split('/')[-2]
        self.output_dir = f"/mnt/data/Dream_ty/Dream_model/ct_infer_img/pics_0903/{self.output_dir}"
        self.exists_or_mkdir(self.output_dir)
        self.keypoint_names = keypoint_names
        # self.debugger = Debugger(opt=opt, dataset=self.trained_dataset)
        
    def run(self, image_or_path_or_tensor, i, json_path, meta={}, is_final=False):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, track_time, tot_time, display_time = 0, 0, 0, 0
        # self.debugger.clear()
        start_time = time.time()
        
        # read image
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            # 我们会在inference的外层使用cv2读入
            # 所以这里会是np.ndarray
            # 原来的paper是按照bbox来划分中心与非中心的，这样就很讨厌，我不能这么稿
            image = image_or_path_or_tensor 
        elif type(image_or_path_or_tensor) == type (''): 
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image'][0].numpy()
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True
        
        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        output_detections = []
        
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                # not prefetch testing or demo
                # 因为我们的未预处理过，所以会先调用pre_process函数
                images, meta = self.pre_process(image, scale, meta)
            else:
                # prefetch testing
                images = pre_processed_images['images'][scale][0]
                meta = pre_processed_images['meta'][scale]
                meta = {k: v.numpy()[0] for k, v in meta.items()}
                if 'pre_dets' in pre_processed_images['meta']:
                    meta['pre_dets'] = pre_processed_images['meta']['pre_dets']
                if 'cur_dets' in pre_processed_images['meta']:
                    meta['cur_dets'] = pre_processed_images['meta']['cur_dets']
          
            images = images.to(self.opt.device, non_blocking=self.opt.non_block_test)
        
            # initializing tracker
            pre_hms, pre_inds = None, None
            if self.opt.tracking:
                # initialize the first frame
                if self.pre_images is None:
                    print('Initialize tracking!')
                    self.pre_images = images
                    self.tracker.init_track(
                        meta['pre_dets'] if 'pre_dets' in meta else [])
                if self.opt.pre_hm:
                    # render input heatmap from tracker status
                    # pre_inds is not used in the current version.
                    # We used pre_inds for learning an offset from previous image to
                    # the current image.
                    # print('pre_hm', meta)
                    if i == 0: 
                        pre_hms, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
                    
                    elif i == 1:
                        _, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
                        pre_hms, _ = self._get_initial_gt_inputs(self.pre_json_path, meta)
                    else:
                        _, pre_inds = self._get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
                        pre_hms = self._get_further_dt_inputs(self.detected_kps, meta, with_hm=True)
#                        pre_hms, _ = self._get_initial_gt_inputs(self.pre_json_path, meta)
                        
#                        pre_hms, _ = self._get_initial_gt_inputs(self.pre_json_path, meta)
                       
#                    pre_hms = torch.zeros(1, 1, 480, 480).to(self.opt.device)
                        
#                    if i == 0:
#                        print('i', i) 
#                        print('pre_hms', pre_hms)
#                        print('pre_inds', pre_inds)

#                     pre_hms, pre_inds = self._get_additional_inputs(
#                        self.tracker.tracks, meta, with_hm=not self.opt.zero_pre_hm)
                                   
            pre_process_time = time.time()
            pre_time += pre_process_time - scale_start_time
            
            # Visualize heatmaps for debugging
#            pre_hm_to_img = dream.image_proc.image_from_belief_map(pre_hms[0][0])
#            self.exists_or_mkdir(self.output_dir)
#            pre_hm_to_img.save(os.path.join(self.output_dir, "{}_pre_hm.png".format(i)))
#           
#            if i > 0:
#                pre_gt_hm_to_img = dream.image_proc.image_from_belief_map(pre_gt_hms[0][0]) 
#                pre_gt_hm_to_img.save(os.path.join(self.output_dir, "{}_pre_gt.png".format(i)))
###            print('images.shape', images.shape)
###            print('images.dtype', images.dtype)
            prev_images_to_img = dream.image_proc.image_from_tensor(self.pre_images[0])
#            prev_images_to_img.save(os.path.join(self.output_dir, "{}_init_gt_img.png".format(i)))
##            # print('shape', self.pre_images[0].shape)
            prev_images_to_img_norm = dream.image_proc.image_from_tensor(self.pre_images[0]/225)
            output_dirs = {head : os.path.join(self.output_dir, f"{head}_overlay") for head in ["whole", "gt", "dt"]}
            
            
            
 
            # run the network
            # output: the output feature maps, only used for visualizing
            # dets: output tensors after extracting peaks
            output, dets, forward_time = self.process(
              images, self.pre_images, pre_hms, pre_inds, return_time=True)
            net_time += forward_time - pre_process_time
            decode_time = time.time()
            dec_time += decode_time - forward_time
#            output_detections.append(deepcopy(dets))
#            for key in dets:
#                try:
#                    print(key, dets[key].shape)
#                except:
#                    print(key, dets[key])
            # print('scores', dets['scores'])
            # print('xs', dets['xs'])
            
            # convert the cropped and 4x downsampled output coordinate system
            # back to the input image coordinate system
            result = self.post_process(dets, meta, scale)
            post_process_time = time.time()
            post_time += post_process_time - decode_time

            detections.append(result)
            # if self.opt.debug >= 2:
            #     self.debug(
            #     self.debugger, images, result, output, scale, 
            #     pre_images=self.pre_images if not self.opt.no_pre_img else None, 
            #     pre_hms=pre_hms)
            
        # merge multi-scale testing results
        results = self.merge_outputs(detections)
#        output_results = self.merge_outputs(output_detections)
        # print('results', results)
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        
        if self.opt.tracking:
            # public detection mode in MOT challenge
            public_det = meta['cur_dets'] if self.opt.public_det else None
            # add tracking id to results
            results = self.tracker.step(results, public_det)
            self.pre_images = images

        tracking_time = time.time()
        track_time += tracking_time - end_time
        tot_time += tracking_time - start_time

        # if self.opt.debug >= 1:
        #     self.show_results(self.debugger, image, results)
        self.cnt += 1

        show_results_time = time.time()
        display_time += show_results_time - end_time
        
        # return results and run time
        ret = {'results': results, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time, 'merge': merge_time, 'track': track_time,
                'display': display_time}
        if self.opt.save_video:
            try:
                # return debug image for saving video
                ret.update({'generic': self.debugger.imgs['generic']})
            except:
                pass
                
        self.detected_kps = self._get_final_kps(results)
        self.pre_json_path = json_path
#        self._get_overlay_imgs(prev_images_to_img_norm, self.detected_kps, json_path, meta, output_dirs, i)
#        self.detected_kps_output = self._get_final_kps_output(results)
#        self._get_overlay_output_imgs(self.detected_kps_output, meta, json_path, output_dirs, i) 
        # print('detected_kps', self.detected_kps)
        # print('meta', meta)
        # print('shape', prev_images_to_img_norm.size) 
        
        
        if not is_final: 
            return ret
        else:
            return ret, self.detected_kps
    
    def _transform_scale(self, image, scale=1):
      '''
        Prepare input image in different testing modes.
          Currently support: fix short size/ center crop to a fixed size/ 
          keep original resolution but pad to a multiplication of 32
      '''
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      if self.opt.fix_short > 0:
          if height < width:
              inp_height = self.opt.fix_short
              inp_width = (int(width / height * self.opt.fix_short) + 63) // 64 * 64
          else:
              inp_height = (int(height / width * self.opt.fix_short) + 63) // 64 * 64
              inp_width = self.opt.fix_short
          c = np.array([width / 2, height / 2], dtype=np.float32)
          s = np.array([width, height], dtype=np.float32)
      elif self.opt.fix_res:
          inp_height, inp_width = self.opt.input_h, self.opt.input_w
          c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
          s = max(height, width) * 1.0
          # s = np.array([inp_width, inp_height], dtype=np.float32)
      else:
          inp_height = (new_height | self.opt.pad) + 1
          inp_width = (new_width | self.opt.pad) + 1
          c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
          s = np.array([inp_width, inp_height], dtype=np.float32)
      resized_image = cv2.resize(image, (new_width, new_height))
      return resized_image, c, s, inp_width, inp_height, height, width
  
    def pre_process(self, image, scale, input_meta={}):
      '''
      Crop, resize, and normalize image. Gather meta data for post processing 
        and tracking.
      '''
      resized_image, c, s, inp_width, inp_height, height, width = \
        self._transform_scale(image)
      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
      out_height =  inp_height // self.opt.down_ratio
      out_width =  inp_width // self.opt.down_ratio
      trans_output = get_affine_transform(c, s, 0, [out_width, out_height])

      inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
      inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

      images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
      if self.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
      images = torch.from_numpy(images)
      meta = {'calib': np.array(input_meta['calib'], dtype=np.float32) \
               if 'calib' in input_meta else \
               self._get_default_calib(width, height)}
      meta.update({'c': c, 's': s, 'height': height, 'width': width,
              'out_height': out_height, 'out_width': out_width,
              'inp_height': inp_height, 'inp_width': inp_width,
              'trans_input': trans_input, 'trans_output': trans_output})
      if 'pre_dets' in input_meta:
        meta['pre_dets'] = input_meta['pre_dets']
      if 'cur_dets' in input_meta:
        meta['cur_dets'] = input_meta['cur_dets']
      return images, meta
   
    def _trans_bbox(self, bbox, trans, width, height):
      '''
      Transform bounding boxes according to image crop.
      '''
      bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
      bbox[:2] = affine_transform(bbox[:2], trans)
      bbox[2:] = affine_transform(bbox[2:], trans)
      bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
      bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
      return bbox
  
    def _get_additional_inputs(self, dets, meta, with_hm=True):
      '''
      Render input heatmap from previous trackings.
      '''
      # 这里是需要改掉的，因为我们没有bbox
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 

      output_inds = []
      for det in dets:
          if det['score'] < self.opt.pre_thresh or det['active'] == 0:
              continue
#          bbox = self._trans_bbox(det['bbox'], trans_input, inp_width, inp_height)
#          bbox_out = self._trans_bbox(
#            det['bbox'], trans_output, out_width, out_height)
#          h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
          ct_det = affine_transform(det["ct_wreg"], trans_input)
          ct_det[0] = np.clip(ct_det[0], 0, inp_width-1)
          ct_det[1] = np.clip(ct_det[1], 0, inp_height-1)
          ct_det_out = affine_transform(det["ct_wreg"], trans_output)
          ct_det_out[0] = np.clip(ct_det_out[0], 0, out_width-1)
          ct_det_out[1] = np.clip(ct_det_out[1], 0, out_height-1)
          
          radius = 4
          ct = np.array(
          [ct_det[0], ct_det[1]], dtype=np.float32)
          if with_hm:
              dream_draw_umich_gaussian(input_hm[0], ct, radius)
          ct_out = np.array(
            [ct_det_out[0], ct_det_out[1]], dtype=np.int32)
          output_inds.append(ct_out[1] * out_width + ct_out[0])
      if with_hm:
          input_hm = input_hm[np.newaxis]
          if self.opt.flip_test:
            input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
          input_hm = torch.from_numpy(input_hm).to(self.opt.device)
      output_inds = np.array(output_inds, np.int64).reshape(1, -1)
      output_inds = torch.from_numpy(output_inds).to(self.opt.device)
      return input_hm, output_inds

    def _get_further_dt_inputs(self, kps_raw_np, meta, with_hm=True):
      '''
      Render input heatmap from previous trackings.
      '''
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
      kps_raw_list = deepcopy(kps_raw_np.tolist())
      for ct_det in kps_raw_list:
          if -999.999 in kps_raw_list:
              continue
          ct_det = affine_transform(ct_det, trans_input)
          ct_det[0] = np.clip(ct_det[0], 0, inp_width-1)
          ct_det[1] = np.clip(ct_det[1], 0, inp_height-1)
          radius = 4
          ct = np.array(
          [ct_det[0], ct_det[1]], dtype=np.float32)
          if with_hm:
              dream_draw_umich_gaussian(input_hm[0], ct, radius)

      if with_hm:
          input_hm = input_hm[np.newaxis]
          if self.opt.flip_test:
            input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
          input_hm = torch.from_numpy(input_hm).to(self.opt.device)
      
      return input_hm
    
    
    def _get_initial_gt_inputs(self, json_path, meta, with_hm=True):
      '''
      Render input heatmap from previous trackings.
      '''
      # 这里是需要改掉的，因为我们没有bbox
      trans_input, trans_output = meta['trans_input'], meta['trans_output']
      inp_width, inp_height = meta['inp_width'], meta['inp_height']
      out_width, out_height = meta['out_width'], meta['out_height']
      input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
      output_inds = []
      parser = YAML(typ="safe")
      with open(json_path, "r") as f:
          data = parser.load(f.read().replace('\t', ' '))
      data = data[0]
      object_keypoints = data["keypoints"]
      gt_kps_raw = []
      for idx, kp_name in enumerate(self.keypoint_names):
          gt_kps_raw.append(object_keypoints[idx]["projected_location"])
      
      for gt_kp_raw in gt_kps_raw:
          ct_det = affine_transform(gt_kp_raw, trans_input)
          ct_det[0] = np.clip(ct_det[0], 0, inp_width-1)
          ct_det[1] = np.clip(ct_det[1], 0, inp_height-1)
          
          radius = 4
          ct = np.array(
          [ct_det[0], ct_det[1]], dtype=np.float32)
          if with_hm:
              dream_draw_umich_gaussian(input_hm[0], ct, radius)

      if with_hm:
          input_hm = input_hm[np.newaxis]
          if self.opt.flip_test:
            input_hm = np.concatenate((input_hm, input_hm[:, :, :, ::-1]), axis=0)
          input_hm = torch.from_numpy(input_hm).to(self.opt.device)
      
      output_inds = np.array(output_inds, np.int64).reshape(1, -1)
      output_inds = torch.from_numpy(output_inds).to(self.opt.device)
      return input_hm, output_inds

    def _get_final_kps(self, dets):
      '''
      Render input heatmap from previous trackings.
      '''
      dets = deepcopy(dets)
      detected_kps = np.full((self.opt.num_classes, 2), -999.999) 
      cls = {}
      for i in range(1, self.opt.num_classes + 1):
          cls[i] = {"x" : [], "y":[]}
#      for det in dets:
#          # print(det)
#          score, clas, ct_wreg = det["score"], det["class"], det["ct_wreg"].tolist()
#          cls[clas]["x"].append([score, score * ct_wreg[0]])
#          cls[clas]["y"].append([score, score * ct_wreg[1]])
#          
#      for i in range(1, self.opt.num_classes + 1):
#          try:
#              x_np = np.array(cls[i]["x"])
#              y_np = np.array(cls[i]["y"])
#              x_ = np.sum(x_np, axis=0)
#              y_ = np.sum(y_np, axis=0)
#              x_res = x_[1] / (x_[0] + 1e-8)
#              y_res = y_[1] / (y_[0] + 1e-8)
#              detected_kps[i-1] = [x_res, y_res]
#          except:
#              pass
      for det in dets:
          # print('ct_wreg', det['ct_wreg'])
          score, clas, ct_wreg = det["score"], det["class"], det["ct_wreg"].tolist()
          # score, clas, ct_wreg = det["score"], det["class"], det["ct"].tolist() 
          cls[clas]["x"].append([score, ct_wreg[0]])
          cls[clas]["y"].append([score, ct_wreg[1]]) 
      
      for i in range(1, self.opt.num_classes+1):
          try:
              x_list, y_list = cls[i]["x"], cls[i]["y"]
              x_list.sort()
              y_list.sort()
              assert x_list[-1][0] == y_list[-1][0]
              x_res = x_list[-1][1]
              y_res = y_list[-1][1]
              detected_kps[i-1] = [x_res, y_res]
              # print('res', [x_res, y_res])
          except:
              pass
      # print(detected_kps)
      # print('detected_kps', detected_kps)
                  
      return detected_kps
      

    def _get_final_kps_output(self, dets):
      '''
      Render input heatmap from previous trackings.
      '''
      dets = deepcopy(dets)
      detected_kps = np.full((self.opt.num_classes, 6), -999.999) 
      cls = {}
      for i in range(1, self.opt.num_classes + 1):
          cls[i] = {"x" : [], "y":[]}
#      for det in dets:
#          # print(det)
#          score, clas, ct_wreg = det["score"], det["class"], det["ct_wreg"].tolist()
#          cls[clas]["x"].append([score, score * ct_wreg[0]])
#          cls[clas]["y"].append([score, score * ct_wreg[1]])
#          
#      for i in range(1, self.opt.num_classes + 1):
#          try:
#              x_np = np.array(cls[i]["x"])
#              y_np = np.array(cls[i]["y"])
#              x_ = np.sum(x_np, axis=0)
#              y_ = np.sum(y_np, axis=0)
#              x_res = x_[1] / (x_[0] + 1e-8)
#              y_res = y_[1] / (y_[0] + 1e-8)
#              detected_kps[i-1] = [x_res, y_res]
#          except:
#              pass
      for det in dets:
          # print('ct_wreg', det['ct_wreg'])
          score, clas, ct_wreg, ct, reg = det["score"], det["class"], det["ct_wreg_output"].tolist(), det["ct_output"].tolist(), det["reg_output"].tolist()
          cls[clas]["x"].append([score, ct_wreg[0], ct[0], reg[0]])
          cls[clas]["y"].append([score, ct_wreg[1], ct[1], reg[1]])
      
      for i in range(1, self.opt.num_classes+1):
          try:
              x_list, y_list = cls[i]["x"], cls[i]["y"]
              x_list.sort()
              y_list.sort()
              assert x_list[-1][0] == y_list[-1][0]
              x_res, x_ct, x_offset = x_list[-1][1:]
              y_res, y_ct, y_offset = y_list[-1][1:]
              detected_kps[i-1] = [x_res, y_res, x_ct, y_ct, x_offset, y_offset]
              # print('res', [x_res, y_res])
          except:
              pass
      # print(detected_kps)
                  
      return detected_kps
          
   
    def _get_default_calib(self, width, height):
      calib = np.array([[self.rest_focal_length, 0, width / 2, 0], 
                          [0, self.rest_focal_length, height / 2, 0], 
                          [0, 0, 1, 0]])
      return calib
    
    def _get_overlay_imgs(self, prev_img_netin, detected_kps_raw_np, json_path, meta, output_dirs, i):
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32) 
        

        parser = YAML(typ="safe")
        with open(json_path, "r") as f:
            data = parser.load(f.read().replace('\t', ' '))
        data = data[0]
        object_keypoints = data["keypoints"]
        gt_kps_raw = []
        detected_kps_raw = detected_kps_raw_np.tolist()
        overlay_whole_images = []
        overlay_gt_images = []
        overlay_dt_images = []
        
        for idx, kp_name in enumerate(self.keypoint_names):
            gt_kps_raw.append(object_keypoints[idx]["projected_location"])
        
        for idx, (gt_kp_raw, detected_kp_raw) in enumerate(zip(gt_kps_raw, detected_kps_raw)):
            #print('i', i)
            ct_gt = affine_transform(gt_kp_raw, trans_input)
            ct_gt[0] = np.clip(ct_gt[0], 0, inp_width-1)
            ct_gt[1] = np.clip(ct_gt[1], 0, inp_height-1)
            
            ct_dt = affine_transform(detected_kp_raw, trans_input)
            ct_dt[0] = np.clip(ct_dt[0], 0, inp_width-1)
            ct_dt[1] = np.clip(ct_dt[1], 0, inp_height-1)
            
            overlay_whole_image = dream.image_proc.overlay_points_on_image(
                prev_img_netin,
                [ct_gt, ct_dt],
                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )
            
            overlay_gt_image = dream.image_proc.overlay_points_on_image(
                prev_img_netin,
                [ct_gt],
                annotation_color_dot=["green"],
                point_diameter=4,
            )
            
            overlay_dt_image = dream.image_proc.overlay_points_on_image(
                prev_img_netin,
                [ct_dt],
                annotation_color_dot=["red"],
                point_diameter=4,
            )
            
            whole_dir = output_dirs['whole']
            this_whole_dir = os.path.join(whole_dir, f"{idx}_kps", "")
            # print(this_whole_dir)
            self.exists_or_mkdir(this_whole_dir)
            s = str(i).zfill(5)
            overlay_whole_image.save(this_whole_dir + f"{s}.png")
            
            # construct videos
            if i == 29:
                # print('constructing videos')
                video_list = os.listdir(this_whole_dir)
                video_list.sort()
                # print(video_list)
                image = Image.open(this_whole_dir + video_list[0])
                video = cv2.VideoWriter(this_whole_dir + "whole.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2, image.size)
                for j in range(1, len(video_list) + 1):
                    img= cv2.imread(os.path.join(this_whole_dir, video_list[j-1])) # 读取图片
                    video.write(img)
                video.release()
                
            
#            print('whole_image', overlay_whole_image.size)
#            print('gt_image', overlay_gt_image.size) 
#            print('dt_image', overlay_dt_image.size)
            
#            overlay_whole_images.append(overlay_whole_image)
#            overlay_gt_images.append(overlay_gt_image)
#            overlay_dt_images.append(overlay_dt_image)
#            
#        overlay_whole_mosaic = dream.image_proc.mosaic_images(
#            overlay_whole_images, rows=2, cols=4, inner_padding_px=10
#        )
#        overlay_gt_mosaic = dream.image_proc.mosaic_images(
#            overlay_gt_images, rows=2, cols=4, inner_padding_px=10
#        )
#        overlay_dt_mosaic = dream.image_proc.mosaic_images(
#            overlay_dt_images, rows=2, cols=4, inner_padding_px=10
#        )
        
#        overlay_whole_mosaic.save(output_dirs["whole"])
#        overlay_gt_mosaic.save(output_dirs["gt"])
#        overlay_dt_mosaic.save(output_dirs["dt"])
    
    def _get_overlay_output_imgs(self, detected_kps_output_np, meta,json_path, output_dirs, i):
        # print('output_kps', detected_kps_output_np)
        trans_input, trans_output = meta['trans_input'], meta['trans_output']
        inp_width, inp_height = meta['inp_width'], meta['inp_height']
        out_width, out_height = meta['out_width'], meta['out_height']
        
        parser = YAML(typ="safe")
        with open(json_path, "r") as f:
            data = parser.load(f.read().replace('\t', ' '))
        data = data[0]
        object_keypoints = data["keypoints"]
        # gt_kps_output = []    
        gt_kps_raw = [] 
        for idx, kp_name in enumerate(self.keypoint_names):
            gt_kps_raw.append(object_keypoints[idx]["projected_location"])

        
        # print('output_kps_gt', np.stack(gt_kps_output))
        # print("raw_kps_gt", np.stack(gt_kps_raw))
        gt_kps_raw_np = np.stack(gt_kps_raw)
        gt_kps_output_np = dream.utilities.affine_transform_and_clip(gt_kps_raw_np, trans_output, out_width, out_height)
        gt_kps_output = gt_kps_output_np.tolist()
        next_hms = dream.utilities.get_hm(gt_kps_output_np, out_width, out_height)
        next_hms_res = []
        for j in range(next_hms.shape[0]):
            c, r = np.where(next_hms[j] == np.max(next_hms[j]))
            next_hms_res.append([r[0], c[0]])
        # print('output_hms_res', next_hms_res) 
        # print('max difference', np.max(np.stack(gt_kps_output) - np.stack(next_hms_res)))
        
        
            
        output_tensor = torch.ones(3, out_height, out_width).to(self.opt.device)
        output_img = dream.image_proc.image_from_tensor(output_tensor / 255)
        detected_kps_output = detected_kps_output_np[:, :2].tolist()
        overlay_dt_images = []
        
        # print('detected_kps_output', detected_kps_output)
        # print('gt_kps_output', gt_kps_output)
        
        for idx, (detected_kp_output, gt_kp_output) in enumerate(zip(detected_kps_output, gt_kps_output)):
            ct_dt = detected_kp_output
            ct_gt = gt_kp_output
            # print('i', i)
            
            overlay_dt_image = dream.image_proc.overlay_points_on_image(
                output_img,
                [ct_gt, ct_dt],
                annotation_color_dot=["green", "red"],
                point_diameter=4,
            )
#            
#            overlay_dt_image = dream.image_proc.overlay_points_on_image(
#                output_img,
#                [ct_gt],
#                annotation_color_dot=["green"],
#                point_diameter=4,
#            )
            
#            overlay_dt_image = dream.image_proc.overlay_points_on_image(
#                output_img,
#                [ct_dt],
#                annotation_color_dot=["red"],
#                point_diameter=4,
#            )
            
            dt_dir = output_dirs['dt']
            this_dt_dir = os.path.join(dt_dir, f"{idx}_kps_output", "")
            # print(this_whole_dir)
            self.exists_or_mkdir(this_dt_dir)
            s = str(i).zfill(5)
            overlay_dt_image.save(this_dt_dir + f"{s}.png")
            if i == 29:
                # print('constructing videos')
                video_list = os.listdir(this_dt_dir)
                video_list.sort()
                # print(video_list)
                image = Image.open(this_dt_dir + video_list[0])
                video = cv2.VideoWriter(this_dt_dir + "dt.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 2, image.size)
                for j in range(1, len(video_list) + 1):
                    img= cv2.imread(os.path.join(this_dt_dir, video_list[j-1])) # 读取图片
                    video.write(img)
                video.release()
        
        n_kp = gt_kps_output_np.shape[0]
        kps_output_save = np.ones((n_kp * 3, 3))
        kps_output_save[:n_kp, :2] = gt_kps_output_np
        kps_output_save[n_kp:2*n_kp, :2] = detected_kps_output_np[:, 2:4]
        kps_output_save[2*n_kp:, :2] = detected_kps_output_np[:, 4:]
        txt_dir = os.path.join(output_dirs['dt'], "doc", "")
        self.exists_or_mkdir(txt_dir)
        s = str(i).zfill(5)
        txt_dir = txt_dir + f"{s}.txt"
        np.savetxt(txt_dir, kps_output_save)
        
        
        
        
    
    def _sigmoid_output(self, output):
      if 'hm' in output:
          output['hm'] = output['hm'].sigmoid_()
      if 'hm_hp' in output:
          output['hm_hp'] = output['hm_hp'].sigmoid_()
      if 'dep' in output:
          output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
          output['dep'] *= self.opt.depth_scale
      return output

    def _flip_output(self, output):
        average_flips = ['hm', 'wh', 'dep', 'dim']
        neg_average_flips = ['amodel_offset']
        single_flips = ['ltrb', 'nuscenes_att', 'velocity', 'ltrb_amodal', 'reg',
        'hp_offset', 'rot', 'tracking', 'pre_hm']
        for head in output:
            if head in average_flips:
                output[head] = (output[head][0:1] + flip_tensor(output[head][1:2])) / 2
            if head in neg_average_flips:
                flipped_tensor = flip_tensor(output[head][1:2])
                flipped_tensor[:, 0::2] *= -1
                output[head] = (output[head][0:1] + flipped_tensor) / 2
            if head in single_flips:
                output[head] = output[head][0:1]
            # if head == 'hps':
            #     output['hps'] = (output['hps'][0:1] + 
            #     flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
            # if head == 'hm_hp':
            #     output['hm_hp'] = (output['hm_hp'][0:1] + \
            #     flip_lr(output['hm_hp'][1:2], self.flip_idx)) / 2

        return output
      
    def process(self, images, pre_images=None, pre_hms=None,
      pre_inds=None, return_time=False):
      with torch.no_grad():
        torch.cuda.synchronize()
        output = self.model(images, pre_images, pre_hms)[-1]
        output = self._sigmoid_output(output)
        output.update({'pre_inds': pre_inds})
        if self.opt.flip_test:
          output = self._flip_output(output)
        torch.cuda.synchronize()
        forward_time = time.time()
        
        dets = dream_generic_decode(output, K=self.opt.K, opt=self.opt)
        torch.cuda.synchronize()
        for k in dets:
          dets[k] = dets[k].detach().cpu().numpy()
      if return_time:
        return output, dets, forward_time
      else:
        return output, dets
    
    def post_process(self, dets, meta, scale=1):
      dets = dream_generic_post_process(
        self.opt, dets, [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes,
        [meta['calib']], meta['height'], meta['width'])
      self.this_calib = meta['calib']
      
      if scale != 1:
        for i in range(len(dets[0])):
          for k in ['bbox', 'hps']:
            if k in dets[0][i]:
              dets[0][i][k] = (np.array(
                dets[0][i][k], np.float32) / scale).tolist()
      return dets[0]

    def merge_outputs(self, detections):
      assert len(self.opt.test_scales) == 1, 'multi_scale not supported!'
      results = []
      for i in range(len(detections[0])):
        if detections[0][i]['score'] > self.opt.out_thresh:
          results.append(detections[0][i])
      return results
  
    def debug(self, debugger, images, dets, output, scale=1, 
      pre_images=None, pre_hms=None):
      img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      if 'hm_hp' in output:
        pred = debugger.gen_colormap_hp(
          output['hm_hp'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hmhp')

      if pre_images is not None:
        pre_img = pre_images[0].detach().cpu().numpy().transpose(1, 2, 0)
        pre_img = np.clip(((
          pre_img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        debugger.add_img(pre_img, 'pre_img')
        if pre_hms is not None:
          pre_hm = debugger.gen_colormap(
            pre_hms[0].detach().cpu().numpy())
          debugger.add_blend_img(pre_img, pre_hm, 'pre_hm')


    def show_results(self, debugger, image, results):
      debugger.add_img(image, img_id='generic')
      if self.opt.tracking:
        debugger.add_img(self.pre_image_ori if self.pre_image_ori is not None else image, 
          img_id='previous')
        self.pre_image_ori = image
      
      for j in range(len(results)):
        if results[j]['score'] > self.opt.vis_thresh:
          if 'active' in results[j] and results[j]['active'] == 0:
            continue
          item = results[j]
          if ('bbox' in item):
            sc = item['score'] if self.opt.demo == '' or \
              not ('tracking_id' in item) else item['tracking_id']
            sc = item['tracking_id'] if self.opt.show_track_color else sc
            
            debugger.add_coco_bbox(
              item['bbox'], item['class'] - 1, sc, img_id='generic')

          if 'tracking' in item:
            debugger.add_arrow(item['ct'], item['tracking'], img_id='generic')
          
          tracking_id = item['tracking_id'] if 'tracking_id' in item else -1
          if 'tracking_id' in item and self.opt.demo == '' and \
            not self.opt.show_track_color:
            debugger.add_tracking_id(
              item['ct'], item['tracking_id'], img_id='generic')

          if (item['class'] in [1, 2]) and 'hps' in item:
            debugger.add_coco_hp(item['hps'], tracking_id=tracking_id,
              img_id='generic')

      if len(results) > 0 and \
        'dep' in results[0] and 'alpha' in results[0] and 'dim' in results[0]:
        debugger.add_3d_detection(
          image if not self.opt.qualitative else cv2.resize(
            debugger.imgs['pred_hm'], (image.shape[1], image.shape[0])), 
          False, results, self.this_calib,
          vis_thresh=self.opt.vis_thresh, img_id='ddd_pred')
        debugger.add_bird_view(
          results, vis_thresh=self.opt.vis_thresh,
          img_id='bird_pred', cnt=self.cnt)
        if self.opt.show_track_color and self.opt.debug == 4:
          del debugger.imgs['generic'], debugger.imgs['bird_pred']
      if 'ddd_pred' in debugger.imgs:
        debugger.imgs['generic'] = debugger.imgs['ddd_pred']
      if self.opt.debug == 4:
        debugger.save_all_imgs(self.opt.debug_dir, prefix='{}'.format(self.cnt))
      else:
        debugger.show_all_imgs(pause=self.pause)
    

    def reset_tracking(self):
      self.tracker.reset()
      self.pre_images = None
      self.pre_image_ori = None
    
    def exists_or_mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        else:
            return True


























