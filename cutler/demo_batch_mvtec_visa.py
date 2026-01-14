# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/demo/demo.py

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import sys
sys.path.append('./')
sys.path.append('../')
from config import add_cutler_config

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "CutLER detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.10,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    data_dir = '/path/to/mvtec_anomaly_detection'
    save_dir = '../mvtec_cutler_conf_010'
    os.makedirs(f"./{save_dir}", exist_ok=True)

    all_objects = os.listdir(data_dir)

    for cl in all_objects:
        if not os.path.isdir(os.path.join(data_dir, cl)):
            continue
        os.makedirs(f"./{save_dir}/{cl}/", exist_ok=True)
        subfolders = os.listdir(os.path.join(data_dir, cl, 'test'))
        for sf in subfolders:
            test_ims = os.listdir(os.path.join(data_dir, cl, 'test', sf))
            os.makedirs(f"{save_dir}/{cl}/{sf}", exist_ok=True)

            for test_im in test_ims:
                image_path = os.path.join(data_dir, cl, 'test', sf, test_im)
                print(image_path)
                # use PIL, to be consistent with evaluation
                img = read_image(image_path, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        image_path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )
                preds = predictions['instances'].pred_masks.squeeze(0)
                preds = preds.cpu().data.numpy()
                np.savez(f"{save_dir}/{cl}/{sf}/{test_im}_cutler.png", preds)
                
                visualized_output.save(f"{save_dir}/{cl}/{sf}/{test_im}_visual.png")

    