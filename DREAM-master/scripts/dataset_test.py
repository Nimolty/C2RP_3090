from enum import IntEnum

import albumentations as albu
import numpy as np
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as TVTransforms
import numpy as np
from ruamel.yaml import YAML

import dream_geo as dream

#def find_ndds_seq_data_in_dir(
#    input_dir, data_extension="json", image_extension=None, requested_image_types="png",
#):
#    # input_dir = "/mnt/data/Dream_ty/franka_data"
#    input_dir = os.path.expanduser(input_dir)
#    assert os.path.exists(
#    input_dir
#    ), 'Expected path "{}" to exist, but it does not.'.format(input_dir)
#    dirlist = os.listdir(input_dir)
#    
#    found_data = []
#    for each_dir in dirlist:
#        found_data_this_video = []
#        output_dir = os.path.join(input_dir, each_dir)
#        # output_dir = "/mnt/data/Dream_ty/franka_data/xxxxx"
#        image_exts_to_try = ["png", "json"]
#        num_image_exts = []
#        for image_ext in image_exts_to_try:
#            num_image_exts.append(len([f for f in os.listdir(output_dir) \
#                                       if f.endswith(image_ext)]))
#        min_num_image_exts = np.min(num_image_exts)
#        if min_num_image_exts == 0: # 说明啥都没生成好
#            continue
#        idx_min = np.where(num_image_exts == min_num_image_exts)[0]
#        # print("idx_min", idx_min)
#        image_extension = image_exts_to_try[idx_min[0]] # here image_extension is png
#        if len(idx_min) > 1 and min_num_image_exts > 0:
#            print(
#            'Multiple sets of images detected in NDDS dataset with different extensions. Using extension "{}".'.format(
#                image_extension
#            )
#            )
#        else:
#            assert isinstance(
#            image_extension, str
#            ), 'If specified, expected "image_extension" to be a string, but it is "{}".'.format(
#            type(image_extension)
#            )
#        
#        image_full_ext = "." + image_extension # ".png"
#        data_full_ext = "." + "json" # ".json"
#        
#        # Read in json files
#        dir_list = os.listdir(output_dir) 
#        png_paths = [f for f in dir_list if f.endswith(image_full_ext)] # 得到list
#        png_paths.sort()
#        
#        data_filenames = [f for f in dir_list if f.endswith(data_full_ext)]
#        data_filenames.sort()
#        data_filenames = data_filenames[:len(png_paths)] # 要保证两者长度一致且互相对应
#        
#        assert len(png_paths) == len(data_filenames)
#        for png, filename in zip(png_paths, data_filenames):
#            assert png[:4] == filename[:4]
#        
#        data_names = [os.path.join(each_dir, os.path.splitext(f)[0][:4]) for f in data_filenames]
#        data_paths = [os.path.join(output_dir, f) for f in data_filenames]
#        image_paths = [os.path.join(output_dir, f) for f in png_paths]
#        
#        length = len(png_paths)
#        assert length >= 1
#        # 当Lengt为1时
#        if length == 1:
#            this_seq = {}
#            this_seq['prev_frame_name'] = data_names[0]
#            this_seq["prev_frame_img_path"] = image_paths[0]
#            this_seq["prev_frame_data_path"] = data_paths[0]
#            this_seq["next_frame_name"] = data_names[1]
#            this_seq["next_frame_img_path"] = image_paths[1]
#            this_seq["next_frame_data_path"] = data_paths[1]
#            found_data_this_video.append(this_seq)
#        else:
#            for i in range(length-1):
#                this_seq = {}
#                this_seq['prev_frame_name'] = data_names[i]
#                this_seq["prev_frame_img_path"] = image_paths[i]
#                this_seq["prev_frame_data_path"] = data_paths[i]
#                this_seq["next_frame_name"] = data_names[i+1]
#                this_seq["next_frame_img_path"] = image_paths[i+1]
#                this_seq["next_frame_data_path"] = data_paths[i+1]
#                found_data_this_video.append(this_seq)
#        
#        found_data = found_data + found_data_this_video
#        
#        # print('each dir', each_dir)
#        # print(found_data_this_video)
#    print(len(found_data))
#    return found_data # 这里我就单独搞一个camera_config吧，不搞别的了
#
#print(find_ndds_seq_data_in_dir(
#    "/mnt/data/Dream_ty/franka_data/", data_extension="json", image_extension=None, requested_image_types="png",
#))




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
        next_keypoint_positions_wrt_cam_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_cam"])
                ).float()
        
        next_keypoint_positions_wrt_robot_as_tensor = torch.from_numpy(
                np.array(next_keypoints["positions_wrt_robot"])
                ).float()
        
        next_kp_projs_net_output_as_tensor = torch.from_numpy(
                np.array(next_kp_projs_net_output)
                ).float()
        
        sample = {
            'prev_image_rgb_input' : prev_image_rgb_net_input_as_tensor,
            "prev_keypoint_projections_output": prev_kp_projs_net_output_as_tensor,
            "prev_keypoint_positions_wrt_cam": prev_keypoint_positions_wrt_cam_as_tensor,
            "prev_keypoint_positions_wrt_robot" : prev_keypoint_positions_wrt_robot_as_tensor,
            'next_image_rgb_input' : next_image_rgb_net_input_as_tensor,
            "next_keypoint_projections_output": next_kp_projs_net_output_as_tensor,
            "next_keypoint_positions_wrt_cam": next_keypoint_positions_wrt_cam_as_tensor,
            "next_keypoint_positions_wrt_robot" : next_keypoint_positions_wrt_robot_as_tensor,
            "config" : datum}
    
        if self.include_belief_maps:
            prev_belief_maps = dream.image_proc.create_belief_map(
                self.network_output_resolution, prev_kp_projs_net_output_as_tensor
            )
            prev_belief_maps_as_tensor = torch.tensor(prev_belief_maps).float()
            sample["prev_belief_maps"] = prev_belief_maps_as_tensor
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

if __name__ == '__main__':
    from PIL import Image
    input_dir = "/mnt/data/Dream_ty/franka_data_0825/"
    keypoint_names = [
    "Link0",
    "Link1",
    "Link3",
    "Link4", 
    "Link6",
    "Link7",
    "Panda_hand",
    ]
    found_data = dream.utilities.find_ndds_seq_data_in_dir(input_dir)
    
    yaml_parser = YAML(typ="safe")
    with open("/home/zekaiyin/summer_ty/DREAM-master/arch_configs/dream_vgg_q.yaml", "r") as f:
        architecture_config_file = yaml_parser.load(f)
    assert (
        "architecture" in architecture_config_file
    ), 'Expected key "architecture" to exist in the architecture config file, but it does not.'
    architecture_config = architecture_config_file
    print(architecture_config.keys())
    image_normalization = architecture_config["architecture"][
            "image_normalization"
        ]
    image_preprocessing = architecture_config["training"]['config']["image_preprocessing"]
    
    train_dataset = ManipulatorNDDSSeqDataset(
        found_data,
        "Franka_Emika_Panda",
        keypoint_names,
        (400, 400),
        (100, 100),
        image_normalization,
        image_preprocessing,
        include_belief_maps=True,
        augment_data=False,
    )
    
    trainingdata = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True
    )

    targets = iter(trainingdata).next()

    for i, b in enumerate(targets["prev_belief_maps"][0]):
        print(b.shape)
        stack = np.stack([b, b, b], axis=0).transpose(2, 1, 0)
        im = Image.fromarray((stack * 255).astype("uint8"))
        if i == 0:
            im.save("{}.png".format(i))
     















