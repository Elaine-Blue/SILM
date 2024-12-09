from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.task = "single_pose"
        self.dataset = "active"
        self.exp_id = "default"
        self.test = True
        '''
            "level of visualization."
            1: only show the final detection results
            2: show the network output features
            3: use matplot to display"  # useful when lunching training with ipython notebook
            4: save all visualizations to disk
        '''
        self.debug = 0
        self.demo = "" # path to image/ image folders/ video. " 'or "webcam"
        self.load_model = "/root/Demo/checkpoints/movenet.pth" # path to pretrained model
        self.resume = False
        self.gpus = [1]
        self.num_workers = 4
        self.not_cuda_benchmark = False
        self.seed = 317

        # log
        self.print_iter=0 # disable progress bar and print to screen.
        self.hide_data_time = True # not display time during training.
        self.save_all = True # save model to disk every 5 epochs.
        # 帮我将下面所有的self.parser.add_argument()语句改成类别的属性
        self.metric = "loss"
        self.vis_thresh = 0.3
        self.debugger_theme = "white"

        self.arch = "movenet"
        self.froze_backbone = True
        self.head_conv = -1
        self.down_ratio = 4

        self.input_res = -1
        self.input_h = -1
        self.input_w = -1

        # train
        self.lr = 1.25e-4
        self.lr_step = "90,120"
        self.num_epochs = 140
        self.batch_size = 32
        self.master_batch_size = -1
        self.num_iters = -1
        self.val_intervals = 5
        self.trainval = False

        # test
        self.flip_test = False
        self.nms = False
        self.K = 100

        # dataset
        self.not_rand_crop = False
        self.shift = 0.1
        self.scale = 0.4
        self.rotate = 0
        self.flip = 0.5
        self.no_color_aug = False
        self.aug_rot = 0

        # loss
        self.mse_loss = False

        # ctdet
        self.reg_loss = "l1"
        self.hm_weight = 1
        self.off_weight = 1
        
        # multi_pose
        self.hp_weight = 1
        self.hm_hp_weight = 1

    def parse(self, args=""):
        self.gpus_str = self.gpus

        if self.head_conv == -1:  # init default head_conv
            if "dla" in self.arch:
                self.head_conv = 256
            elif "movenet" in self.arch:
                self.head_conv = 96
            else:
                self.head_conv = 64
        self.pad = 127 if "hourglass" in self.arch else 31
        self.num_stacks = 2 if self.arch == "hourglass" else 1

        if self.trainval:
            self.val_intervals = 100000000

        if self.debug > 0:
            self.num_workers = 0
            self.batch_size = 1
            # self.gpus = [self.gpus[0]]
            self.master_batch_size = -1

        # if self.master_batch_size == -1:
        #     self.master_batch_size = self.batch_size // len(self.gpus)
        rest_batch_size = self.batch_size - self.master_batch_size
        self.chunk_sizes = [self.master_batch_size]
        # for i in range(len(self.gpus) - 1):
        #     slave_chunk_size = rest_batch_size // (len(self.gpus) - 1)
        #     if i < rest_batch_size % (len(self.gpus) - 1):
        #         slave_chunk_size += 1
        #     self.chunk_sizes.append(slave_chunk_size)
        # print("training chunk_sizes:", self.chunk_sizes)

        self.root_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        self.data_dir = os.path.join(self.root_dir, "data")
        self.exp_dir = os.path.join(self.root_dir, "exp", self.task)
        self.save_dir = os.path.join(self.exp_dir, self.exp_id)
        self.debug_dir = os.path.join(self.save_dir, "debug")
        print("The output will be saved to ", self.save_dir)

        if self.resume and self.load_model == "":
            model_path = (
                self.save_dir[:-4] if self.save_dir.endswith("TEST") else self.save_dir
            )
            self.load_model = os.path.join(model_path, "model_last.pth")
        return self

    def update_dataset_info_and_set_heads(self, dataset):
        input_h, input_w = dataset.default_resolution
        self.mean, self.std = dataset.mean, dataset.std
        self.num_classes = dataset.num_classes

        # input_h(w): self.input_h overrides self.input_res overrides dataset default
        input_h = self.input_res if self.input_res > 0 else input_h
        input_w = self.input_res if self.input_res > 0 else input_w
        self.input_h = self.input_h if self.input_h > 0 else input_h
        self.input_w = self.input_w if self.input_w > 0 else input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)

        if self.task == "multi_pose":
            assert self.dataset in ["coco_hp"]
            self.flip_idx = dataset.flip_idx
            self.heads = {"hm": self.num_classes, "wh": 2, "hps": 34}
            self.heads.update({"reg": 2})
            self.heads.update({"hm_hp": 17})
            self.heads.update({"hp_offset": 2})
            raise KeyError("The multi_pose is not supported for now.")
        elif self.task == "single_pose":
            self.flip_idx = dataset.flip_idx
            self.heads = {"hm": self.num_classes, "hps": 34, "hm_hp": 17, "hp_offset": 34}
        else:
            assert 0, "task not defined!"
        print("heads", self.heads)
        return self

    def init(self, args=""):
        default_dataset_info = {
            "multi_pose": {
                "default_resolution": [512, 512],
                "num_classes": 1,
                "mean": [0.408, 0.447, 0.470],
                "std": [0.289, 0.274, 0.278],
                "dataset": "coco_hp",
                "num_joints": 17,
                "flip_idx": [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                ],
            },
            "single_pose": {
                "default_resolution": [512, 512],
                "num_classes": 1,
                "mean": [1.0, 1.0, 1.0],
                "std": [1.0, 1.0, 1.0],
                "dataset": "active",
                "num_joints": 17,
                "flip_idx": [
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                ],
            },
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        self.parse(args)
        dataset = Struct(default_dataset_info[self.task])
        self.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(dataset)
        return opt
