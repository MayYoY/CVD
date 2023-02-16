import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as transF
from torch.utils import data
from scipy import io


class VIPL_HR(data.Dataset):
    def __init__(self, config):
        super(VIPL_HR, self).__init__()
        record = pd.read_csv(config.record_path)
        self.config = config

        self.subjects = []
        self.input_paths = []
        self.indices = []
        self.gts = []
        for i in range(len(record)):
            if self.isValid(record, i):
                self.subjects.append(record.loc[i, "video"])
                self.input_paths.append(record.loc[i, "path"])
                self.indices.append(record.loc[i, "idx"])
                self.gts.append(record.loc[i, "beat_num"])

    def isValid(self, record, idx):
        flag = True
        if self.config.folds:
            flag &= record.loc[idx, "fold"] in self.config.folds
        if self.config.tasks:
            flag &= record.loc[idx, "task"] in self.config.tasks
        if self.config.sources:
            flag &= record.loc[idx, "source"] in self.config.sources
        return flag

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img_path1 = self.input_paths[idx] + '/img_rgb.mat'
        img_path2 = self.input_paths[idx] + '/img_yuv.mat'
        feature_map1 = np.array(io.loadmat(img_path1)['img1'])  # RGB
        feature_map2 = np.array(io.loadmat(img_path2)['img2'])  # YUV
        # data augment
        """if self.config.VerticalFlip:
            if random.random() < 0.5:
                feature_map1 = transF.vflip(feature_map1)
                feature_map2 = transF.vflip(feature_map2)"""
        if self.config.trans:
            feature_map1 = self.config.trans(feature_map1)
            feature_map2 = self.config.trans(feature_map2)
        feature_map = torch.cat((feature_map1, feature_map2), dim=0)
        # beat num
        bpm_path = self.input_paths[idx] + '/bpm.mat'
        bpm = io.loadmat(bpm_path)['bpm']
        bpm = bpm.astype('float32')

        fps_path = self.input_paths[idx] + '/fps.mat'
        fps = io.loadmat(fps_path)['fps']
        fps = fps.astype('float32')

        bvp_path = self.input_paths[idx] + '/bvp.mat'
        bvp = io.loadmat(bvp_path)['bvp']
        bvp = bvp.astype('float32')
        bvp = bvp[0]

        gt = torch.tensor(self.gts[idx], dtype=torch.float)  # 由 bvp 计算的 hr

        return feature_map, bpm, fps, bvp, gt, self.subjects[idx]
