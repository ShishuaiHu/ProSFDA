# -*- coding:utf-8 -*-
from torch.utils import data
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk


def load_all_nii(nii_list):
    all_data = list()
    for nii in nii_list:
        nii_sitk = sitk.ReadImage(nii)
        nii_npy = sitk.GetArrayFromImage(nii_sitk)
        for s in range(nii_npy.shape[0]):
            all_data.append(nii_npy[s])
    return all_data


def load_all_nii_unlabeled(nii_list):
    all_data = list()
    for nii in nii_list:
        nii_sitk = sitk.ReadImage(nii)
        nii_npy = sitk.GetArrayFromImage(nii_sitk)
        for s in range(nii_npy.shape[0]):
            if nii_npy[s].max() > 0:
                all_data.append(nii_npy[s])
    return all_data


class GM_labeled_set(data.Dataset):
    def __init__(self, root, img_list, label_list, target_size=(128, 128), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list
        self.label_list = label_list

        self.img_nii_list = [join(self.root, i) for i in self.img_list]
        self.label_nii_list = [join(self.root, i) for i in self.label_list]

        self.img_data_list = load_all_nii(self.img_nii_list)
        self.label_data_list = load_all_nii(self.label_nii_list)

        self.len = len(self.img_data_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_npy = self.img_data_list[item]
        label_npy = self.label_data_list[item]
        mask_npy = np.zeros_like(label_npy)
        mask_npy[label_npy == 1] = 2
        mask_npy[label_npy == 2] = 1
        if self.img_normalize:
            img_npy = img_npy.clip(np.percentile(img_npy, 5), np.percentile(img_npy, 95))
            img_npy = (img_npy - img_npy.mean()) / img_npy.std()
        return np.expand_dims(img_npy, 0).repeat(3, axis=0), mask_npy[np.newaxis], None


class GM_unlabeled_set(data.Dataset):
    def __init__(self, root, img_list, target_size=(128, 128), img_normalize=True):
        super().__init__()
        self.root = root
        self.img_list = img_list

        self.img_nii_list = [join(self.root, i) for i in self.img_list]
        self.img_data_list = load_all_nii_unlabeled(self.img_nii_list)

        self.len = len(self.img_data_list)
        self.target_size = target_size
        self.img_normalize = img_normalize

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img_npy = self.img_data_list[item]
        if self.img_normalize:
            img_npy = img_npy.clip(np.percentile(img_npy, 5), np.percentile(img_npy, 95))
            img_npy = (img_npy - img_npy.mean()) / img_npy.std()
        return np.expand_dims(img_npy, 0).repeat(3, axis=0), None, None
