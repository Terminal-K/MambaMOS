"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np

import random

from .builder import DATASETS
from .defaults import DefaultMultiScansDataset

def points_transform(points, from_pose, to_pose):
  transformation = np.linalg.inv(to_pose).dot(from_pose)
  points = np.hstack((points, np.ones((points.shape[0], 1)))).T

  return transformation.dot(points).T[:, :3]

def load_poses(pose_path):
    """ Load ground truth poses (T_w_cam0) from file.
        Args:
            pose_path: (Complete) filename for the pose file
        Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
    """
    # Read and parse the poses
    poses = []
    try:
        if '.txt' in pose_path:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)['arr_0']

    except FileNotFoundError:
        print('Ground truth poses are not avaialble.')

    return np.array(poses)

def load_calib(calib_path):
    """ Load calibrations (T_cam_velo) from file.
    """
    # Read and parse the calibrations
    T_cam_velo = []
    try:
        with open(calib_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Tr:' in line:
                    line = line.replace('Tr:', '')
                    T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

    except FileNotFoundError:
        print('Calibrations are not avaialble.')

    return np.array(T_cam_velo)

def absoluteFilePaths(directory, selected_file_list=None):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if selected_file_list is None or f.split('.')[0] in selected_file_list:
                yield os.path.abspath(os.path.join(dirpath, f))

@DATASETS.register_module()
class SemanticKITTIMultiScansDataset(DefaultMultiScansDataset):
    def __init__(
        self,
        split="train",
        data_root="data",
        gather_num=6,
        scan_modulation=False,
        transform=None,
        test_mode=False,
        test_cfg=None,
        windows_stride=None,
        loop=1,
        ignore_index=-1,
    ):
        self.gather_num = gather_num
        self.scan_modulation = scan_modulation
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            gather_num=gather_num,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

        self.windows_stride = windows_stride

    def get_pose_data(self, pose_file, calib_file):
        # load poses
        poses = np.array(load_poses(pose_file))
        inv_frame0 = np.linalg.inv(poses[0])

        # load calibrations
        T_cam_velo = load_calib(calib_file)
        T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
        T_velo_cam = np.linalg.inv(T_cam_velo)

        # convert kitti poses from camera coord to LiDAR coord
        new_poses = []
        for pose in poses:
            new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))

        return np.array(new_poses)

    def filter_dataset(self, seq_list):
        with open(os.path.join(os.path.dirname(__file__), "./train_split_dynamic_pointnumber.txt")) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}
        # num_dict = {}
        for line in lines:
            if line != '':
                # seq, fid, moving_points_num = line.split()
                seq, fid, moving_points_num = line.split()
                if int(seq) in seq_list:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]
        return pending_dict

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            road_train=[30, 31, 32, 33, 34, 40],
            road_test=[35, 36, 37, 38, 39, 41],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        self.poses = {}
        pending_dict = self.filter_dataset(seq_list)
        for seq in seq_list:
            seq = str(seq).zfill(2)
            # seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_folder = os.path.join(self.data_root, "sequences", seq)
            seq_pose_file = os.path.join(seq_folder, "poses.txt")
            seq_calib_file = os.path.join(seq_folder, "calib.txt")

            self.poses[seq] = self.get_pose_data(pose_file=seq_pose_file, calib_file=seq_calib_file)

            if self.split == "train":
                data_list += absoluteFilePaths(os.path.join(seq_folder, "velodyne"), selected_file_list=pending_dict[seq])
                print(f"Filter few static frame --- Seq {seq} drops "
                      f"{len(os.listdir(os.path.join(seq_folder, 'velodyne'))) - len(pending_dict[seq])} : "
                      f"{len(os.listdir(os.path.join(seq_folder, 'velodyne')))} -> {len(pending_dict[seq])}")
            else:
                data_list += absoluteFilePaths(os.path.join(seq_folder, "velodyne"))
            # seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            # data_list += [
            #     os.path.join(seq_folder, "velodyne", file) for file in seq_files
            # ]
        return data_list

    def get_multi_data(self, idx):
        cur_data_path = self.data_list[idx % len(self.data_list)]

        multi_scan_path, gather_coord, gather_strength, gather_segment = [], [], [], []
        seq, _, file_name = cur_data_path.split('/')[-3:]
        gather_seq = [seq for _ in range(self.gather_num)]

        cur_scan_index = int(file_name.split('.')[0])

        tn = []

        modulation = 1
        if self.scan_modulation:
            scan_modulation_prob = random.random()
            if 0.5 < scan_modulation_prob <= 0.75:
                modulation = 2
            elif scan_modulation_prob > 0.75:
                modulation = 3
            if self.split != "train":
                modulation = 3

        for i, seq in enumerate(gather_seq):
            last_scan_index = cur_scan_index - modulation * i
            last_scan_index = max(0, last_scan_index)  # 5, 4, 3, 2, 1, 0, 0, 0
            scan_path = cur_data_path.replace(cur_data_path.split("/")[-1], str(last_scan_index).zfill(6) + ".bin")

            with open(scan_path, "rb") as b:
                scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)

                coord = points_transform(scan[:, :3], from_pose=self.poses[seq][last_scan_index],
                                         to_pose=self.poses[seq][cur_scan_index])

            strength = scan[:, -1].reshape([-1, 1])

            label_file = scan_path.replace("velodyne", "labels").replace(".bin", ".label")
            if os.path.exists(label_file):
                with open(label_file, "rb") as a:
                    segment = np.fromfile(a, dtype=np.int32).reshape(-1) & 0xFFFF
                    segment = np.vectorize(self.learning_map.__getitem__)(
                        segment
                    ).astype(np.int32)
            else:
                segment = np.zeros(scan.shape[0]).astype(np.int32)

            gather_coord.append(coord)
            gather_strength.append(strength)
            gather_segment.append(segment)

            multi_scan_path.append(scan_path)
            tn.append(np.ones_like(segment) * i)

        data_dict = dict(coord=np.concatenate(gather_coord), strength=np.concatenate(gather_strength),
                         segment=np.concatenate(gather_segment), tn=np.expand_dims(np.concatenate(tn), axis=1))

        return data_dict

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"

        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            9: 1,
            10: 2,  # "car"
            11: 2,  # "bicycle"
            13: 2,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 2,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 2,  # "truck"
            20: 2,  # "other-vehicle"
            30: 2,  # "person"
            31: 2,  # "bicyclist"
            32: 2,  # "motorcyclist"
            40: 1,  # "road"
            44: 1,  # "parking"
            48: 1,  # "sidewalk"
            49: 1,  # "other-ground"
            50: 1,  # "building"
            51: 1,  # "fence"
            52: 1,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 1,  # "lane-marking" to "road" ---------------------------------mapped
            70: 1,  # "vegetation"
            71: 1,  # "trunk"
            72: 1,  # "terrain"
            80: 1,  # "pole"
            81: 1,  # "traffic-sign"
            99: 1,  # "other-object" to "unlabeled" ----------------------------mapped
            250: 2,
            251: 3,
            252: 3,  # "moving-car" to "car" ------------------------------------mapped
            253: 3,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 3,  # "moving-person" to "person" ------------------------------mapped
            255: 3,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 3,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 3,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 3,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }

        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            1: 9,
            2: 250,
            3: 251,
        }

        return learning_map_inv
