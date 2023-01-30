# -*- coding:utf-8 -*-
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from PIL import Image


def normalize_img_to_0255(img):
    return (img-img.min())/(img.max()-img.min()) * 255


def main():
    base = r"dir base for predictions of unlabeled target domain images"
    out_base = r'output dir'
    csv_file = r"csv file path of unlabeled target domain images"

    with open(csv_file, 'r') as f:
        f_list = f.read().split('\n')[1:-1]

    c_list = [c_f.split('/')[-1] for c_f in f_list]

    maybe_mkdir_p(out_base)
    channel_0_file = join(base, 'data_channel0.nii.gz')
    channel_1_file = join(base, 'data_channel1.nii.gz')
    channel_2_file = join(base, 'data_channel2.nii.gz')
    cup_file = join(base, 'output_cup.nii.gz')
    disc_file = join(base, 'output_disc.nii.gz')

    channel_0_sitk = sitk.ReadImage(channel_0_file)
    channel_1_sitk = sitk.ReadImage(channel_1_file)
    channel_2_sitk = sitk.ReadImage(channel_2_file)
    cup_sitk = sitk.ReadImage(cup_file)
    disc_sitk = sitk.ReadImage(disc_file)

    channel_0_npy = sitk.GetArrayFromImage(channel_0_sitk)
    channel_1_npy = sitk.GetArrayFromImage(channel_1_sitk)
    channel_2_npy = sitk.GetArrayFromImage(channel_2_sitk)
    cup_npy = sitk.GetArrayFromImage(cup_sitk)
    disc_npy = sitk.GetArrayFromImage(disc_sitk)

    case_number = channel_0_npy.shape[-1]

    for i in range(case_number):
        case_img = np.zeros((512, 512, 3))
        case_seg = np.zeros((512, 512))
        case_img[:, :, 0] = normalize_img_to_0255(channel_0_npy[:, :, i]).astype(np.uint8)
        case_img[:, :, 1] = normalize_img_to_0255(channel_1_npy[:, :, i]).astype(np.uint8)
        case_img[:, :, 2] = normalize_img_to_0255(channel_2_npy[:, :, i]).astype(np.uint8)
        case_seg[disc_npy[:, :, i] > 0.5] = 255
        case_seg[cup_npy[:, :, i] > 0.5] = 128

        case_img_f = Image.fromarray(case_img.astype(np.uint8)).resize((800, 800)).rotate(270).transpose(Image.FLIP_LEFT_RIGHT)
        case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((800, 800), resample=Image.NEAREST).rotate(270).transpose(Image.FLIP_LEFT_RIGHT)

        case_img_f.save(join(out_base, c_list[i]))
        case_seg_f.save(join(out_base, c_list[i].replace('.tif', '-1.tif')))

    with open(join(out_base, 'self-training.csv'), 'w') as f:
        f.write('image,mask\n')
        for i in f_list:
            f.write('{},{}\n'.format(i, i.replace('RIGA/', 'RIGA-mask/')))


if __name__ == '__main__':
    main()
