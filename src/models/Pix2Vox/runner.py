#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import os
import sys

from pprint import pprint

from tensorboardX import SummaryWriter

from config import cfg
from core.train import train_net
from core.test import test_net
from utils.data_loaders import DatasetType


def test_model(model_type, test_dataset: str, batch_size: int,
               mvs_taxonomy_file: str, results_file_name=None, weights_path=None, dataset_type=DatasetType.TEST,
               n_views: int = 1, save_results_to_file: bool = True, show_voxels: bool=False, path_to_times_csv=None):
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)

    cfg.DATASET.TEST_DATASET = test_dataset
    cfg.DATASETS.MVS.TAXONOMY_FILE_PATH = mvs_taxonomy_file
    cfg.CONST.BATCH_SIZE = batch_size
    cfg.CONST.N_VIEWS_RENDERING = n_views

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    if weights_path:
        cfg.CONST.WEIGHTS = weights_path

    if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
        test_net(cfg, model_type, dataset_type, test_writer=SummaryWriter(), save_results_to_file=save_results_to_file,
                 results_file_name=results_file_name, show_voxels=show_voxels, path_to_times_csv=path_to_times_csv)
    else:
        logging.error('Please specify the file path of checkpoint.')
        sys.exit(2)


def train_model(model_type, train_dataset: str, test_dataset: str,
                shapenet_ratio: int, batch_size: int,
                mvs_taxonomy_file: str):
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)

    cfg.DATASET.TRAIN_DATASET = train_dataset
    cfg.DATASET.TEST_DATASET = test_dataset
    cfg.DATASETS.MVS.TAXONOMY_FILE_PATH = mvs_taxonomy_file
    cfg.CONST.SHAPENET_RATIO = shapenet_ratio
    cfg.CONST.BATCH_SIZE = batch_size

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    train_net(cfg, model_type)
