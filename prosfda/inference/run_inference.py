# -*- coding:utf-8 -*-
import argparse
from prosfda.utils.file_utils import gen_random_str
from batchgenerators.utilities.file_and_folder_operations import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="unet", required=False,
                        help='Model name.')
    parser.add_argument('--chk', default="model_final.model", required=False,
                        help='Checkpoint name.')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], required=False,
                        help='Device id.')
    parser.add_argument('--log_folder', required=True,
                        help='Log folder.')
    parser.add_argument('--tag', default="{}".format(gen_random_str()), required=False,
                        help='Run identifier.')
    parser.add_argument('--inference_tag', default="all", required=False,
                        help='Inference tag.')
    parser.add_argument('--patch_size', type=int, nargs='+', default=[512, 512], required=False,
                        help='patch size.')
    parser.add_argument('-r', '--root', required=True,
                        help='root folder.')
    parser.add_argument('--ts_csv', nargs='+',
                        required=True, help='test csv file.')
    parser.add_argument('--pretrained_model', required=False,
                        help='pretrained model path.')
    parser.add_argument('--prompt_model_path', required=False,
                        help='domain-aware prompt model path.')

    args = parser.parse_args()
    model_name = args.model
    chk_name = args.chk
    gpu = tuple(args.gpu)
    log_folder = args.log_folder
    tag = args.tag
    log_folder = join(log_folder, model_name+'_'+tag)
    patch_size = tuple(args.patch_size)
    root_folder = args.root
    ts_csv = tuple(args.ts_csv)
    inference_tag = args.inference_tag
    pretrained_model_path = args.pretrained_model
    prompt_model_path = args.prompt_model_path

    if model_name == 'UNet':
        from prosfda.inference.inference_nets.inference_unet import inference
        inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, inference_tag)
    elif model_name == 'PLS':
        from prosfda.inference.inference_nets.inference_pls import inference
        inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, pretrained_model_path, inference_tag)
    elif model_name == 'FAS':
        from prosfda.inference.inference_nets.inference_fas import inference
        inference(chk_name, gpu, log_folder, patch_size, root_folder, ts_csv, prompt_model_path, inference_tag)
    else:
        print('No model named "{}"!'.format(model_name))
        return


if __name__ == '__main__':
    main()
