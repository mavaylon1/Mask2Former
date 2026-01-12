# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F
import h5py
import os

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

__all__ = ["MaskFormerSemanticDatasetKDMapper"]


class MaskFormerSemanticDatasetKDMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        logits_file,
        kd_temp,
        kd_weight,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        
        self.logits_file = logits_file
        self.kd_temp = kd_temp
        self.kd_weight = kd_weight
        
        if self.logits_file is not None:
            self.h5 = h5py.File(self.logits_file, "r")
        # breakpoint()
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        
        if not cfg.KD:
            logits_file = None
            kd_temp = None
        else:
            logits_file = cfg.KD.FILE
            kd_temp = cfg.KD.TEMP
            kd_weight = cfg.KD.WEIGHT
            
        

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "logits_file": logits_file,
            "kd_temp": kd_temp,
            "kd_weight": kd_weight
        }
        return ret

    def __call__(self, dataset_dict):
        assert self.is_train, "KD mapper should only be used for training"

        dataset_dict = copy.deepcopy(dataset_dict)

        # -------------------------------------------------
        # Image + GT processing (normal Mask2Former logic)
        # -------------------------------------------------
        filename = os.path.basename(dataset_dict["file_name"]).replace(".jpg", "")

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        sem_seg_gt = utils.read_image(
            dataset_dict.pop("sem_seg_file_name")
        ).astype("double")

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)

        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        image = torch.as_tensor(image.transpose(2, 0, 1).copy())
        sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            h, w = image.shape[-2:]
            pad_h = (self.size_divisibility - h % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - w % self.size_divisibility) % self.size_divisibility
            image = F.pad(image, (0, pad_w, 0, pad_h), value=128)
            sem_seg_gt = F.pad(sem_seg_gt, (0, pad_w, 0, pad_h), value=self.ignore_label)

        dataset_dict["image"] = image
        dataset_dict["sem_seg"] = sem_seg_gt.long()

        # -------------------------------------------------
        # Instances (unchanged)
        # -------------------------------------------------
        instances = Instances(image.shape[-2:])
        classes = torch.unique(sem_seg_gt)
        classes = classes[classes != self.ignore_label]
        instances.gt_classes = classes

        masks = [(sem_seg_gt == c) for c in classes]
        if len(masks):
            instances.gt_masks = BitMasks(torch.stack(masks)).tensor
        else:
            instances.gt_masks = torch.zeros((0, *sem_seg_gt.shape))

        dataset_dict["instances"] = instances
        
        dataset_dict["teacher_probs"] = torch.from_numpy(
            self.h5[filename]["semantic_probs"][:]
        ).float()
        
        dataset_dict['kd_temp'] = self.kd_temp
        dataset_dict['kd_weight'] = self.kd_weight

        return dataset_dict


