#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import os
import sys

from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net


def main(is_train: bool, model_type, weights_path=None):
    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.DEBUG)
    # Print config
    print('Use config:')
    pprint(cfg)

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    if weights_path:
        cfg.CONST.WEIGHTS = weights_path
    # Start train/test process
    if is_train:
        train_net(cfg, model_type)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            test_net(cfg, model_type)
        else:
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)
