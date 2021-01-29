# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import logging
import subprocess
import time

import numpy as np
import torch
import torch.backends.cudnn
import torch.utils.data

import utils.binvox_rw as br
import utils.data_loaders
import utils.data_transforms
import utils.helpers
from models.decoder import Decoder
from models.encoder import Encoder
from models.merger import Merger
from models.model_types import Pix2VoxTypes
from models.refiner import Refiner
from settings import VIEWVOX_EXE
from utils.average_meter import AverageMeter
from utils.results_saver import save_test_results_to_csv, save_times_to_csv


def test_net(cfg,
             model_type,
             dataset_type,
             results_file_name,
             epoch_idx=-1,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None,
             save_results_to_file=False,
             show_voxels=False,
             path_to_times_csv=None):
    if model_type == Pix2VoxTypes.Pix2Vox_A or model_type == Pix2VoxTypes.Pix2Vox_Plus_Plus_A:
        use_refiner = True
    else:
        use_refiner = False

    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            dataset_type, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
            batch_size=1,
            num_workers=cfg.CONST.NUM_WORKER,
            pin_memory=True,
            shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg, model_type)
        decoder = Decoder(cfg, model_type)
        if use_refiner:
            refiner = Refiner(cfg)
        merger = Merger(cfg, model_type)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            if use_refiner:
                refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        logging.info('Loading weights from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if use_refiner:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = AverageMeter()
    if use_refiner:
        refiner_losses = AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    if use_refiner:
        refiner.eval()
    merger.eval()

    samples_names = []
    edlosses = []
    rlosses = []
    ious_dict = {}
    for iou_threshold in cfg.TEST.VOXEL_THRESH:
        ious_dict[iou_threshold] = []

    if path_to_times_csv is not None:
        n_view_list = []
        times_list = []

    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.helpers.var_or_cuda(rendering_images)
            ground_truth_volume = utils.helpers.var_or_cuda(ground_truth_volume)

            if path_to_times_csv is not None:
                start_time = time.time()

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if use_refiner and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss

            if path_to_times_csv is not None:
                end_time = time.time()
                n_view_list.append(rendering_images.size()[1])
                times_list.append(end_time - start_time)

            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            if use_refiner:
                refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                sample_iou.append((intersection / union).item())

                ious_dict[th].append((intersection / union).item())

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            # Append generated volumes to TensorBoard
            if show_voxels:
                with open("model.binvox", "wb") as f:
                    v = br.Voxels(torch.ge(generated_volume, 0.2).float().cpu().numpy()[0], (32, 32, 32), (0, 0, 0), 1,
                                  "xyz")
                    v.write(f)

                subprocess.run([VIEWVOX_EXE, "model.binvox"])

                with open("model.binvox", "wb") as f:
                    v = br.Voxels(ground_truth_volume.cpu().numpy()[0], (32, 32, 32), (0, 0, 0), 1, "xyz")
                    v.write(f)

                subprocess.run([VIEWVOX_EXE, "model.binvox"])

            # Print sample loss and IoU
            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                         (sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                          refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

            samples_names.append(sample_name)
            edlosses.append(encoder_loss.item())
            if use_refiner:
                rlosses.append(refiner_loss.item())

    if save_results_to_file:
        save_test_results_to_csv(samples_names, edlosses, rlosses, ious_dict, path_to_csv=results_file_name)

    if path_to_times_csv is not None:
        save_times_to_csv(times_list, n_view_list, path_to_csv=path_to_times_csv)

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    for mi in mean_iou:
        print('%.4f' % mi, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        if use_refiner:
            test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
            test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou
