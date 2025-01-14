# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os.path as osp
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import argparse
from os import path as osp
from dataset.av2_converter import AV2_Converter

def av2_data_prep(root_path,
                  out_dir,
                  workers=16):
    """Prepare data related to AV2 dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 5
    """
    # train_converter = AV2_Converter(root_path, out_dir, 'train', workers)
    # train_converter.convert()
    val_converter = AV2_Converter(root_path, out_dir, 'val', workers)
    val_converter.convert()
    
parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('--dataset', metavar='av2', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='argoverse2 data path', # './data/av2'
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
# parser.add_argument(
#     '--max-sweeps',
#     type=int,
#     default=5,
#     required=True,
#     help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--num-points',
    type=int,
    default=-1,
    help='Number of points to sample for indoor datasets.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='pkl_file_save_path', # '/root/autodl-tmp/av2_data/trainval_test'
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='av2')
parser.add_argument(
    '--workers', type=int, default=64, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'av2':
        av2_data_prep(
            root_path=args.root_path,
            out_dir=args.out_dir,
            workers=args.workers
        )
   