# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
# from lib.utils.image import flip, color_aug
# from lib.utils.image import get_affine_transform, affine_transform
from lib.utils.image import gaussian_radius, draw_umich_gaussian
import copy
import cv2

import dream_geo as dream

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Debug mode:
# 0: no debug mode
# 1: light debug
# 2: heavy debug
class ManipulatorNDDSDatasetDebugLevels(IntEnum):
    # No debug information
    NONE = 0
    # Minor debug information, passing of extra info but not saving to disk
    LIGHT = 1
    # Heavy debug information, including saving data to disk
    HEAVY = 2
    # Interactive debug mode, not intended to be used for actual training
    INTERACTIVE = 3


class ManipulatorNDDSDataset(TorchDataset):
    def __init__(
        self,
        ndds_dataset,
        manipulator_name,
        keypoint_names,
        network_input_resolution,
        network_output_resolution,
        image_normalization,
        image_preprocessing,
        augment_data=False,
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"],
    ):
        # Read in the camera intrinsics
        self.ndds_dataset_data = ndds_dataset[0]
        self.ndds_dataset_config = ndds_dataset[1]
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.network_input_resolution = network_input_resolution
        self.network_output_resolution = network_output_resolution
        self.augment_data = augment_data

        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps

        self.debug_mode = debug_mode

        assert (
            isinstance(image_normalization, dict) or not image_normalization
        ), 'Expected image_normalization to be either a dict specifying "mean" and "stdev", or None or False to specify no normalization.'

        # Image normalization
        # Basic PIL -> tensor without normalization, used for visualizing the net input image
        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )

        if image_normalization:
            assert (
                "mean" in image_normalization and len(image_normalization["mean"]) == 3
            ), 'When image_normalization is a dict, expected key "mean" specifying a 3-tuple to exist, but it does not.'
            assert (
                "stdev" in image_normalization
                and len(image_normalization["stdev"]) == 3
            ), 'When image_normalization is a dict, expected key "stdev" specifying a 3-tuple to exist, but it does not.'

            self.tensor_from_image_tform = TVTransforms.Compose(
                [
                    TVTransforms.ToTensor(),
                    TVTransforms.Normalize(
                        image_normalization["mean"], image_normalization["stdev"]
                    ),
                ]
            )
        else:
            # Use the PIL -> tensor tform without normalization if image_normalization isn't specified
            self.tensor_from_image_tform = self.tensor_from_image_no_norm_tform

        assert (
            image_preprocessing in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES
        ), 'Image preprocessing type "{}" is not recognized.'.format(
            image_preprocessing
        )
        self.image_preprocessing = image_preprocessing

    def __len__(self):
        return len(self.ndds_dataset_data)

    def __getitem__(self, index):

        # Parse this datum
        datum = self.ndds_dataset_data[index]
        image_rgb_path = datum["image_paths"]["rgb"]

        # Extract keypoints from the json file
        data_path = datum["data_path"]
        if self.include_ground_truth:
            keypoints = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, self.keypoint_names
            )
        else:
            # Generate an empty 'keypoints' dict
            keypoints = dream.utilities.load_keypoints(
                data_path, self.manipulator_name, []
            )

        # Load image and transform to network input resolution -- pre augmentation
        image_rgb_raw = PILImage.open(image_rgb_path).convert("RGB")
        image_raw_resolution = image_rgb_raw.size

        # Do image preprocessing, including keypoint conversion
        image_rgb_before_aug = dream.image_proc.preprocess_image(
            image_rgb_raw, self.network_input_resolution, self.image_preprocessing
        )
        kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
            keypoints["projections"],
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing,
        )

        # Handle data augmentation
        if self.augment_data:
            augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            data_to_aug = {
                "image": np.array(image_rgb_before_aug),
                "keypoints": kp_projs_before_aug,
            }
            augmented_data = augmentation(**data_to_aug)
            image_rgb_net_input = PILImage.fromarray(augmented_data["image"])
            kp_projs_net_input = augmented_data["keypoints"]
            #print('augment')
        else:
            image_rgb_net_input = image_rgb_before_aug
            kp_projs_net_input = kp_projs_before_aug

        assert (
            image_rgb_net_input.size == self.network_input_resolution
        ), "Expected resolution for image_rgb_net_input to be equal to specified network input resolution, but they are different."

        # Now convert keypoints at network input to network output for use as the trained label
        kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
            kp_projs_net_input,
            self.network_input_resolution,
            self.network_output_resolution,
        )

        # Convert to tensor for output handling
        # This one goes through image normalization (used for inference)
        image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
            image_rgb_net_input
        )

        # This one is not (used for net input overlay visualizations - hence "viz")
        image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
            image_rgb_net_input
        )

        # Convert keypoint data to tensors - use float32 size
        keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
            np.array(keypoints["positions_wrt_cam"])
        ).float()
        
        kp_projs_net_output_as_tensor = torch.from_numpy(
            np.array(kp_projs_net_output)
        ).float()


        if keypoints['trans'] != [] and keypoints['rot_quat'] != []:
            trans_gt_as_tensor = torch.from_numpy(
                np.array(keypoints["trans"])
            ).float()
            rot_gt_as_tensor = torch.from_numpy(
                np.array(keypoints["rot_quat"])
            ).float()
            keypoint_positions_as_tensor = torch.from_numpy(
            np.array(keypoints["positions"])
            ).float()
            keypoint_positions_as_tensor = keypoint_positions_as_tensor.squeeze()
             
        # Construct output sample
            sample = {
                "image_rgb_input": image_rgb_net_input_as_tensor,
                ######################
                "keypoint_projections_output": kp_projs_net_output_as_tensor,
                "keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
                "trans_keypoint_positions": keypoint_positions_as_tensor,
                "trans_gt": trans_gt_as_tensor,
                "rot_gt": rot_gt_as_tensor,
                ######################
                "config": datum,
            }
        else:
            sample = {
                'image_rgb_input' : image_rgb_net_input_as_tensor,
                "keypoint_projections_output": kp_projs_net_output_as_tensor,
                "keypoint_positions": keypoint_positions_wrt_cam_as_tensor,
                "config": datum,
            }
        #print('key_positions:', keypoint_positions_as_tensor.shape)

        # Generate the belief maps directly
        if self.include_belief_maps:
            belief_maps = dream.image_proc.create_belief_map(
                self.network_output_resolution, kp_projs_net_output_as_tensor
            )
            belief_maps_as_tensor = torch.tensor(belief_maps).float()
            sample["belief_maps"] = belief_maps_as_tensor
            #print('belief_maps')

        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["LIGHT"]:
            kp_projections_as_tensor = torch.from_numpy(
                np.array(keypoints["projections"])
            ).float()
            sample["keypoint_projections_raw"] = kp_projections_as_tensor
            kp_projections_input_as_tensor = torch.from_numpy(kp_projs_net_input).float()

            sample["keypoint_projections_input"] = kp_projections_input_as_tensor
            image_raw_resolution_as_tensor = torch.tensor(image_raw_resolution).float()
            sample["image_resolution_raw"] = image_raw_resolution_as_tensor
            sample["image_rgb_input_viz"] = image_rgb_net_input_viz_as_tensor
            #print('debug_mode')

        # TODO: same as LIGHT debug, but also saves to disk
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["HEAVY"]:
            pass

        # Display to screen
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["INTERACTIVE"]:
            # Ensure that the points are consistent with the image transformations
            # The overlaid points on both image should be consistent, despite image transformations
            debug_image_raw = dream.image_proc.overlay_points_on_image(
                image_rgb_raw, keypoints["projections"], self.keypoint_names
            )
            debug_image_raw.show()

            debug_image = dream.image_proc.overlay_points_on_image(
                image_rgb_net_input, kp_projs_net_input, self.keypoint_names
            )
            debug_image.show()

            # Also show that the output resolution data are consistent
            image_rgb_net_output = image_rgb_net_input.resize(
                self.network_output_resolution, resample=PILImage.BILINEAR
            )
            debug_image_rgb_net_output = dream.image_proc.overlay_points_on_image(
                image_rgb_net_output, kp_projs_net_output, self.keypoint_names
            )
            debug_image_rgb_net_output.show()

            if self.include_belief_maps:
                for kp_idx in range(len(self.keypoint_names)):
                    belief_map_kp = dream.image_proc.image_from_belief_map(
                        belief_maps_as_tensor[kp_idx]
                    )
                    belief_map_kp.show()

                    belief_map_kp_upscaled = belief_map_kp.resize(
                        self.network_input_resolution, resample=PILImage.BILINEAR
                    )
                    image_rgb_net_output_belief_blend = PILImage.blend(
                        image_rgb_net_input, belief_map_kp_upscaled, alpha=0.5
                    )
                    image_rgb_net_output_belief_blend_overlay = dream.image_proc.overlay_points_on_image(
                        image_rgb_net_output_belief_blend,
                        [kp_projs_net_input[kp_idx]],
                        [self.keypoint_names[kp_idx]],
                    )
                    image_rgb_net_output_belief_blend_overlay.show()
            
            print('interactive')
            # This only works if the number of workers is zero
            input("Press Enter to continue...")
        
        #print(sample['keypoint_projections_output'].max(), sample['keypoint_projections_output'].min())
        return sample

class ManipulatorNDDSSeqDataset(TorchDataset):
    def __init__(
        self,
        ndds_seq_dataset,
        manipulator_name, # 这个咱们还得自己搞一个，模仿Dream的
        keypoint_names,
        network_input_resolution,
        network_output_resolution,
        image_normalization,
        image_preprocessing,
        augment_data=False,
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"]
    ):
        self.ndds_seq_dataset_data = ndds_seq_dataset
        # self.ndds_seq_dataset_config = dataset_config
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.network_input_resolution = network_input_resolution
        self.network_output_resolution = network_output_resolution
        self.augment_data = augment_data
        
        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        
        self.debug_mode = debug_mode

        assert (
            isinstance(image_normalization, dict) or not image_normalization
        ), 'Expected image_normalization to be either a dict specifying "mean" and "stdev", or None or False to specify no normalization.'
        
        # Image normalization
        # Basic PIL -> tensor without normalization, used for visualizing the net input image
        self.tensor_from_image_no_norm_tform = TVTransforms.Compose(
            [TVTransforms.ToTensor()]
        )

        if image_normalization:
            assert (
                "mean" in image_normalization and len(image_normalization["mean"]) == 3
            ), 'When image_normalization is a dict, expected key "mean" specifying a 3-tuple to exist, but it does not.'
            assert (
                "stdev" in image_normalization
                and len(image_normalization["stdev"]) == 3
            ), 'When image_normalization is a dict, expected key "stdev" specifying a 3-tuple to exist, but it does not.'

            self.tensor_from_image_tform = TVTransforms.Compose(
                [
                    TVTransforms.ToTensor(),
                    TVTransforms.Normalize(
                        image_normalization["mean"], image_normalization["stdev"]
                    ),
                ]
            )
        else:
            # Use the PIL -> tensor tform without normalization if image_normalization isn't specified
            self.tensor_from_image_tform = self.tensor_from_image_no_norm_tform
        
        assert (
            image_preprocessing in dream.image_proc.KNOWN_IMAGE_PREPROC_TYPES
        ), 'Image preprocessing type "{}" is not recognized.'.format(
            image_preprocessing
        )
        self.image_preprocessing = image_preprocessing
    
    def __len__(self):
        return len(self.ndds_seq_dataset_data)
    
    def __getitem__(self, index):
        # 得到一个dict,里面有{prev_frame_name; prev_frame_img_path, 
        # prev_frame_data_path, next_frame_name, next_frame_img_path, next_frame_data_path}
        
        # Parse this datum
        datum = self.ndds_seq_dataset_data[index]
        prev_frame_name = datum["prev_frame_name"]
        prev_frame_img_path = datum["prev_frame_img_path"]
        prev_frame_data_path = datum["prev_frame_data_path"]
        next_frame_name = datum["next_frame_name"]
        next_frame_img_path = datum["next_frame_img_path"]
        next_frame_data_path = datum["next_frame_data_path"]
        
        if self.include_ground_truth:
            prev_keypoints = dream.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
            next_keypoints = dream.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
        else:
            prev_keypoints = dream.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, [])
            next_keypoints = dream.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, [])
        
        # load iamge and transform to network input resolution --pre augmentation
        prev_image_rgb_raw = PILImage.open(prev_frame_img_path).convert("RGB")
        next_image_rgb_raw = PILImage.open(next_frame_img_path).convert("RGB")
        image_raw_resolution = prev_image_rgb_raw.size
        
        # Do image preprocessing, including keypoint conversion
        prev_image_rgb_before_aug = dream.image_proc.preprocess_image(
            prev_image_rgb_raw, self.network_input_resolution, self.image_preprocessing
            )
        prev_kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
            prev_keypoints["projections"],
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing
            )
        next_image_rgb_before_aug = dream.image_proc.preprocess_image(
            next_image_rgb_raw, self.network_input_resolution, self.image_preprocessing
            )
        next_kp_projs_before_aug = dream.image_proc.convert_keypoints_to_netin_from_raw(
            next_keypoints["projections"],
            image_raw_resolution,
            self.network_input_resolution,
            self.image_preprocessing
            )
        
        # Handle data augmentation
        if self.augment_data:
            prev_augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            prev_data_to_aug = {
                "image": np.array(prev_image_rgb_before_aug),
                "keypoints": prev_kp_projs_before_aug,
            }
            prev_augmented_data = prev_augmentation(**data_to_aug)
            prev_mage_rgb_net_input = PILImage.fromarray(prev_augmented_data["image"])
            prev_kp_projs_net_input = prev_augmented_data["keypoints"]
            next_augmentation = albu.Compose(
                [
                    albu.GaussNoise(),
                    albu.RandomBrightnessContrast(brightness_by_max=False),
                    albu.ShiftScaleRotate(rotate_limit=15),
                ],
                p=1.0,
                keypoint_params={"format": "xy", "remove_invisible": False},
            )
            next_data_to_aug = {
                "image": np.array(next_image_rgb_before_aug),
                "keypoints": next_kp_projs_before_aug,
            }
            next_augmented_data = next_augmentation(**data_to_aug)
            next_mage_rgb_net_input = PILImage.fromarray(next_augmented_data["image"])
            next_kp_projs_net_input = next_augmented_data["keypoints"]
        else:
            prev_image_rgb_net_input = prev_image_rgb_before_aug
            prev_kp_projs_net_input = prev_kp_projs_before_aug
            next_image_rgb_net_input = next_image_rgb_before_aug
            next_kp_projs_net_input = next_kp_projs_before_aug
        
        assert (
            prev_image_rgb_net_input.size == self.network_input_resolution
        )
        
        # Now convert keypoints at network input to network output for use as the trained label
        prev_kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
                prev_kp_projs_net_input,
                self.network_input_resolution,
                self.network_output_resolution,
            )
        next_kp_projs_net_output = dream.image_proc.convert_keypoints_to_netout_from_netin(
                next_kp_projs_net_input,
                self.network_input_resolution,
                self.network_output_resolution,
            )
        
        # Convert to tensor for ouput handling
        # This one goes through image normalization (used for inference)
        prev_image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
                prev_image_rgb_net_input
            )
        next_image_rgb_net_input_as_tensor = self.tensor_from_image_tform(
                next_image_rgb_net_input
            )
        
        # This one is not used for net input overlay visualizaiotns --hence viz
        prev_image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
                prev_image_rgb_net_input
            )
        next_image_rgb_net_input_viz_as_tensor = self.tensor_from_image_no_norm_tform(
                next_image_rgb_net_input
            )
        
        #Convert keypoint data to tensors -use float32 size 
        prev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_cam"])
                ).float()
        
        prev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_robot"])
                ).float() 
        prev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(prev_kp_projs_net_output)
                ).float()
        prev_kp_projs_net_output_as_tensor_int = prev_kp_projs_net_output_as_tensor.int()
        
        
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output)
                ).float()
        next_kp_projs_net_output_as_tensor_int = dream.utilities.make_int(next_kp_projs_net_output_as_tensor, self.network_output_resolution)
        
         
        
        
        sample = {
            'prev_image_rgb_input' : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,
            'next_image_rgb_input' : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,
            "next_keypoint_projections_output_int": next_kp_projs_net_output_as_tensor_int, 
            "reg" : next_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "tracking" : prev_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "config" : datum}
    
        if self.include_belief_maps:
            prev_belief_maps = dream.image_proc.create_belief_map(
                self.network_output_resolution, prev_kp_projs_net_output_as_tensor
            )
            prev_belief_maps_as_tensor = torch.tensor(prev_belief_maps).float()
            sample["prev_belief_maps"] = prev_belief_maps_as_tensor
            
            prev_kp_projs_net_input_as_tensor = torch.from_numpy(np.array(prev_kp_projs_net_input)).float()
            prev_belief_maps_as_input_resolution = dream.image_proc.create_belief_map_as_whole(
                self.network_input_resolution, prev_kp_projs_net_input_as_tensor
            )
            prev_belief_maps_as_input_resolution_as_tensor = torch.tensor(prev_belief_maps_as_input_resolution).float()
            sample['prev_belief_maps_as_input_resolution'] = prev_belief_maps_as_input_resolution_as_tensor 
            
            next_belief_maps = dream.image_proc.create_belief_map(
                self.network_output_resolution, next_kp_projs_net_output_as_tensor
            )
            next_belief_maps_as_tensor = torch.tensor(next_belief_maps).float()
            sample["next_belief_maps"] = next_belief_maps_as_tensor
        
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["LIGHT"]:
            prev_kp_projections_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["projections"])
            ).float()
            sample["prev_keypoint_projections_raw"] = prev_kp_projections_as_tensor
            prev_kp_projections_input_as_tensor = torch.from_numpy(prev_kp_projs_net_input).float()

            sample["prev_keypoint_projections_input"] = prev_kp_projections_input_as_tensor
            prev_image_raw_resolution_as_tensor = torch.tensor(prev_image_raw_resolution).float()
            sample["prev_image_resolution_raw"] = prev_image_raw_resolution_as_tensor
            sample["prev_image_rgb_input_viz"] = prev_image_rgb_net_input_viz_as_tensor
            #print('debug_mode')
            next_kp_projections_as_tensor = torch.from_numpy(
                np.array(next_keypoints["projections"])
            ).float()
            sample["next_keypoint_projections_raw"] = next_kp_projections_as_tensor
            next_kp_projections_input_as_tensor = torch.from_numpy(next_kp_projs_net_input).float()

            sample["next_keypoint_projections_input"] = next_kp_projections_input_as_tensor
            next_image_raw_resolution_as_tensor = torch.tensor(next_image_raw_resolution).float()
            sample["next_image_resolution_raw"] = next_image_raw_resolution_as_tensor
            sample["next_image_rgb_input_viz"] = next_image_rgb_net_input_viz_as_tensor

        # TODO: same as LIGHT debug, but also saves to disk
        if self.debug_mode >= ManipulatorNDDSDatasetDebugLevels["HEAVY"]:
            pass

        return sample
        
class CenterTrackSeqDataset(TorchDataset):
    def __init__(
        self,
        ndds_seq_dataset,
        manipulator_name, # 这个咱们还得自己搞一个，模仿Dream的
        keypoint_names,
        opt, 
        mean, 
        std, 
        include_ground_truth=True,
        include_belief_maps=False,
        debug_mode=ManipulatorNDDSDatasetDebugLevels["NONE"]
    ): 
        self.ndds_seq_dataset_data = ndds_seq_dataset
        # self.ndds_seq_dataset_config = dataset_config
        self.manipulator_name = manipulator_name
        self.keypoint_names = keypoint_names
        self.opt = opt
        self.input_w, self.input_h = self.opt.input_w, self.opt.input_h
        self.output_w, self.output_h = self.opt.output_w, self.opt.output_h
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        
        # If include_belief_maps is specified, include_ground_truth must also be
        # TBD: revisit better way of passing inputs, maybe to make one argument instead of two
        
        if include_belief_maps:
            assert (
                include_ground_truth
            ), 'If "include_belief_maps" is True, "include_ground_truth" must also be True.'
        self.include_ground_truth = include_ground_truth
        self.include_belief_maps = include_belief_maps
        
        self.debug_mode = debug_mode
    
    def __len__(self):
        return len(self.ndds_seq_dataset_data)
    
    def __getitem__(self, index):
        # 得到一个dict,里面有{prev_frame_name; prev_frame_img_path, 
        # prev_frame_data_path, next_frame_name, next_frame_img_path, next_frame_data_path}
        
        # Parse this datum
        datum = self.ndds_seq_dataset_data[index]
        prev_frame_name = datum["prev_frame_name"]
        prev_frame_img_path = datum["prev_frame_img_path"]
        prev_frame_data_path = datum["prev_frame_data_path"]
        next_frame_name = datum["next_frame_name"]
        next_frame_img_path = datum["next_frame_img_path"]
        next_frame_data_path = datum["next_frame_data_path"]
        
        if self.include_ground_truth:
            prev_keypoints = dream.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
            next_keypoints = dream.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, self.keypoint_names)
        else:
            prev_keypoints = dream.utilities.load_seq_keypoints(prev_frame_data_path, \
                            self.manipulator_name, [])
            next_keypoints = dream.utilities.load_seq_keypoints(next_frame_data_path, \
                            self.manipulator_name, [])
        
        # load iamge and transform to network input resolution --pre augmentation
        prev_image_rgb_raw = cv2.imread(prev_frame_img_path)
        next_image_rgb_raw = cv2.imread(next_frame_img_path)
        assert prev_image_rgb_raw.shape == next_image_rgb_raw.shape
        
        # 对rgb_raw进行augment
        # print('prev_image_rgb_raw', prev_image_rgb_raw.shape)
        height, width, _ = prev_image_rgb_raw.shape
        c = np.array([prev_image_rgb_raw.shape[1] / 2., prev_image_rgb_raw.shape[0] / 2.], dtype=np.float32) # (width/2, height/2)
        s = max(prev_image_rgb_raw.shape[0], prev_image_rgb_raw.shape[1]) * 1.0
        aug_s, rot = 1.0, 0 # 让rot为0是表示这里不需要旋转，因为我们需要x3d的信息，所以不做旋转
        c, aug_s = dream.utilities._get_aug_param(c, s, width, height)
        s = s * aug_s
        
        # 得到了raw_to_input的仿射变换， raw_to_output的仿射变换
        # trans_input, trans_output均为2x3的矩阵，且rot为0的情况下只有scale和translation
        trans_input = dream.utilities.get_affine_transform(
        c, s, rot, [self.input_w, self.input_h]) 
        trans_output = dream.utilities.get_affine_transform(
        c, s, rot, [self.output_w, self.output_h])
        prev_image_rgb_net_input = dream.utilities._get_input(prev_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std) # 3 x H x W
        next_image_rgb_net_input = dream.utilities._get_input(next_image_rgb_raw, trans_input, self.input_w, self.input_h, self.mean, self.std)
        assert prev_image_rgb_net_input.shape == next_image_rgb_net_input.shape
        assert prev_image_rgb_net_input.shape == (3, self.input_h, self.input_w)
        prev_image_rgb_net_input_as_tensor = torch.from_numpy(prev_image_rgb_net_input).float()
        next_image_rgb_net_input_as_tensor = torch.from_numpy(next_image_rgb_net_input).float()
        
        # 现在需要转换keypoints到output上，即在self.output_h, self.output_w上的keypoints\
        prev_kp_projs_raw_np = np.array(prev_keypoints["projections"], dtype=np.float32)
        next_kp_projs_raw_np = np.array(next_keypoints["projections"], dtype=np.float32)
        prev_kp_projs_net_output_np = dream.utilities.affine_transform_and_clip(prev_kp_projs_raw_np, trans_output, self.output_w, self.output_h)
        next_kp_projs_net_output_np = dream.utilities.affine_transform_and_clip(next_kp_projs_raw_np, trans_output, self.output_w, self.output_h)
        
        # Convert keypoint data to tensors -use float32 size 
        prev_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_cam"])
                ).float()
        
        prev_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(prev_keypoints["positions_wrt_robot"])
                ).float() 
                
        prev_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(prev_kp_projs_net_output_np)
                ).float()
        # prev_kp_projs_net_output_as_tensor_int = prev_kp_projs_net_output_as_tensor.int()
        
        
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output_np) 
                ).float()
        next_kp_projs_net_output_as_tensor_int = dream.utilities.make_int(next_kp_projs_net_output_as_tensor, [self.output_w, self.output_h])
        next_kp_projs_net_output_int_np = next_kp_projs_net_output_as_tensor_int.numpy()
        
        sample = {
            "prev_image_raw_path" : prev_frame_img_path, 
            "prev_image_rgb_input" : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,
            "next_image_raw_path" : next_frame_img_path, 
            "next_image_rgb_input" : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,
            "next_keypoint_projections_output_int": next_kp_projs_net_output_as_tensor_int, 
            "reg" : next_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "tracking" : prev_kp_projs_net_output_as_tensor - next_kp_projs_net_output_as_tensor_int,
            "config" : datum}
        
        
#        print('next_kp_projs_net_output_as_tensor', next_kp_projs_net_output_as_tensor)
#        print("next_keypoint_projections_output_int", next_kp_projs_net_output_as_tensor_int)
#        print("prev_keypoint_projections_output", prev_kp_projs_net_output_as_tensor)
#        print('reg', sample['reg'])
#        print('tracking', sample["tracking"])
    
        if self.include_belief_maps:
#            prev_belief_maps = dream.utilities.get_hm(prev_kp_projs_net_output_np, self.output_w, self.output_h)
#            prev_belief_maps_as_tensor = torch.from_numpy(prev_belief_maps).float()
#            sample["prev_belief_maps"] = prev_belief_maps_as_tensor
            
            prev_belief_maps_as_whole_np = dream.utilities.get_prev_hm(prev_kp_projs_raw_np, trans_input,self.input_w, self.input_h, hm_disturb = self.opt.hm_disturb, lost_disturb=self.opt.lost_disturb, fp_disturb=self.opt.fp_disturb) 
            prev_belief_maps_as_whole_as_tensor = torch.from_numpy(prev_belief_maps_as_whole_np).float()
            sample["prev_belief_maps_as_input_resolution"] = prev_belief_maps_as_whole_as_tensor
            
            # print('next_kp_projs_net_output_as_tensor', next_kp_projs_net_output_as_tensor)
            # print('next_kp_projs_net_output_int_np', next_kp_projs_net_output_int_np)
            next_belief_maps = dream.utilities.get_hm(next_kp_projs_net_output_int_np, self.output_w, self.output_h)
            next_belief_maps_as_tensor = torch.from_numpy(next_belief_maps).float()
            sample["next_belief_maps"] = next_belief_maps_as_tensor
            


        return sample

if __name__ == "__main__":
    from PIL import Image

    # beliefs = CreateBeliefMap((100,100),[(50,50),(-1,-1),(0,50),(50,0),(10,10)])
    # for i,b in enumerate(beliefs):
    #     print(b.shape)
    #     stack = np.stack([b,b,b],axis=0).transpose(2,1,0)
    #     im = Image.fromarray((stack*255).astype('uint8'))
    #     im.save('{}.png'.format(i))

    path = "/home/sbirchfield/data/FrankaSimpleHomeDR20k/"
    # path = '/home/sbirchfield/data/FrankaSimpleMPGammaDR105k/'

    keypoint_names = [
        "panda_link0",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link6",
        "panda_link7",
        "panda_hand",
    ]

    found_data = dream.utilities.find_ndds_data_in_dir(path)
    train_dataset = ManipulatorNDDSDataset(
        found_data,
        "panda",
        keypoint_names,
        (400, 400),
        (100, 100),
        include_belief_maps=True,
        augment_data=True,
    )
    trainingdata = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
    )

    targets = iter(trainingdata).next()

    for i, b in enumerate(targets["belief_maps"][0]):
        # print(b.shape)
        stack = np.stack([b, b, b], axis=0).transpose(2, 1, 0)
        im = Image.fromarray((stack * 255).astype("uint8"))
        im.save("{}.png".format(i))
