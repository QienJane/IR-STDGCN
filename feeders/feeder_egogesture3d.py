import json
import random
import math

import numpy as np
from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self, data_path, label_path, repeat=1, random_choose=False, random_shift=False, random_move=False,
                 window_size=8, normalization=False, debug=False, use_mmap=True):
        """
        EgoGesture3D dataset feeder.

        Args:
            data_path: Data mode ('joint', 'bone', 'motion')
            label_path: Path containing 'train' or 'val' to determine split
            repeat: Number of times to repeat the dataset
            random_choose: If True, randomly choose a portion of the input sequence
            random_shift: If True, randomly pad zeros at the beginning or end of sequence
            random_move: If True, apply random movement transformation
            window_size: The length of the output sequence (default: 8)
            normalization: If True, normalize input sequence
            debug: If True, only use the first 100 samples
            use_mmap: If True, use mmap mode to load data
        """
        self.egogesture3d_root = 'data/egogesture3d/egogesture3d_jsons/'

        if 'val' in label_path:
            self.train_val = 'val'
            with open(self.egogesture3d_root + 'val_samples.json', 'r') as f:
                json_file = json.load(f)
            self.data_dict = json_file
            self.flag = 'val_jsons/'
        else:
            self.train_val = 'train'
            with open(self.egogesture3d_root + 'train_samples.json', 'r') as f:
                json_file = json.load(f)
            self.data_dict = json_file
            self.flag = 'train_jsons/'

        # Bone pairs for EgoGesture3D (42 nodes: 21 left hand + 21 right hand)
        # Single hand structure: wrist (0) connected to finger bases, then finger chains
        single_hand_bone = [
            (1, 0), (2, 1), (3, 2), (4, 3),      # Thumb: 0->1->2->3->4
            (5, 0), (6, 5), (7, 6), (8, 7),      # Index: 0->5->6->7->8
            (9, 0), (10, 9), (11, 10), (12, 11), # Middle: 0->9->10->11->12
            (13, 0), (14, 13), (15, 14), (16, 15), # Ring: 0->13->14->15->16
            (17, 0), (18, 17), (19, 18), (20, 19), # Pinky: 0->17->18->19->20
            (0, 0),  # Self-loop for wrist
        ]
        # Left hand bones (nodes 0-20)
        left_hand_bone = single_hand_bone.copy()
        # Right hand bones (nodes 21-41)
        right_hand_bone = [(i + 21, j + 21) for (i, j) in single_hand_bone]
        self.bone = left_hand_bone + right_hand_bone

        self.load_data()
        self.data_path = data_path
        self.repeat = repeat
        self.window_size = window_size

        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info['label']))

        self.debug = debug
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.normalization = normalization
        self.use_mmap = use_mmap

    def load_data(self):
        """Load skeleton data from JSON files."""
        self.data = []  # data: T N C
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.egogesture3d_root + self.flag + file_name + '.json', 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            self.data.append(value)

    def random_translation(self, ske_data):
        """Apply random translation to skeleton data."""
        translate = np.eye(3)
        t_x = random.uniform(-0.01, 0.01)
        t_y = random.uniform(-0.01, 0.01)
        t_z = random.uniform(-0.01, 0.01)

        translate[0, 0] = translate[0, 0] + t_x
        translate[1, 1] = translate[1, 1] + t_y
        translate[2, 2] = translate[2, 2] + t_z

        data = np.dot(ske_data, translate)
        return data

    def rand_view_transform(self, X, agx, agy, s):
        """Apply random view transformation (rotation and scale)."""
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == 'train':
            # Apply random transformations during training
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0, 0, :]  # Use wrist as center
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            # Normalize to [-1, 1]
            min_val = np.min(scalerValue, axis=0)
            max_val = np.max(scalerValue, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            scalerValue = (scalerValue - min_val) / range_val
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 42, 3))

            data = np.zeros((self.window_size, 42, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            # Random sampling
            random_idx = random.sample(list(np.arange(length)) * 100, self.window_size)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]
        else:
            # No random transformations during validation/testing
            agx = 0
            agy = 0
            s = 1.0

            center = value[0, 0, :]  # Use wrist as center
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            # Normalize to [-1, 1]
            min_val = np.min(scalerValue, axis=0)
            max_val = np.max(scalerValue, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            scalerValue = (scalerValue - min_val) / range_val
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 42, 3))

            data = np.zeros((self.window_size, 42, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            # Linear sampling
            idx = np.linspace(0, length - 1, self.window_size).astype(int)
            data[:, :, :] = value[idx, :, :]  # T, V, C

        # Apply bone mode transformation
        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)  # T N C
            for bone_idx in range(len(self.bone)):
                data_bone[:, self.bone[bone_idx][0], :] = data[:, self.bone[bone_idx][0], :] - data[:, self.bone[bone_idx][1], :]
            data = data_bone

        # Apply motion mode transformation
        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion

        # Transpose to C, T, N format
        data = np.transpose(data, (2, 0, 1))
        C, T, N = data.shape
        data = np.reshape(data, (C, T, N, 1))  # C T N 1

        return data, label, index

    def top_k(self, score, top_k):
        """Calculate top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
