from pprint import pprint
from typing import Iterable, List

import albumentations as A
import av
import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn as nn
from torchvision import transforms


def extract_meta_av(path: str) -> (int, int, int):
    try:
        video = av.open(path)
        video_stream = video.streams.video[0]
        return video_stream.height, video_stream.width, video_stream.frames
    except av.AVError as e:
        print('Error while reading file: {}'.format(path))
        print(e)
        return 0, 0, 0
    except IndexError as e:
        print('Error while processing file: {}'.format(path))
        print(e)
        return 0, 0, 0


def extract_meta_cv(path: str) -> (int, int, int):
    try:
        vid = cv2.VideoCapture(path)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        return height, width, num_frames
    except Exception as e:
        print('Error while reading file: {}'.format(path))
        print(e)
        return 0, 0, 0


def adapt_bb(frame_height: int, frame_width: int, bb_height: int, bb_width: int, left: int, top: int, right: int,
             bottom: int) -> (
        int, int, int, int):
    x_ctr = (left + right) // 2
    y_ctr = (bottom + top) // 2
    new_top = max(y_ctr - bb_height // 2, 0)
    new_bottom = min(new_top + bb_height, frame_height)
    new_left = max(x_ctr - bb_width // 2, 0)
    new_right = min(new_left + bb_width, frame_width)
    return new_left, new_top, new_right, new_bottom


def extract_bb(frame: Image.Image, bb: Iterable, size: int) -> Image.Image:
    left, top, right, bottom = bb

    bb_width = int(right) - int(left)
    bb_height = int(bottom) - int(top)
    bb_to_desired_ratio = min(size / bb_height, size / bb_width) if (bb_width > 0 and bb_height > 0) else 1.
    bb_width = int(size / bb_to_desired_ratio)
    bb_height = int(size / bb_to_desired_ratio)

    left, top, right, bottom = adapt_bb(
        frame.height, frame.width, bb_height, bb_width, left, top, right, bottom
    )

    face = frame.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
    return face


def showimage(img_tensor: torch.Tensor):
    topil = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0, ], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        transforms.ToPILImage()
    ])
    plt.figure()
    plt.imshow(topil(img_tensor))
    plt.show()


def make_train_tag(net_class: nn.Module, patch_size: int, traindb: List[str], seed: int,suffix: str, debug: bool,):
    tag_params = dict(net=net_class.__name__,
                      traindb='-'.join(traindb),
                      size=patch_size,
                      seed=seed
                      )
    print('Parameters')
    pprint(tag_params)
    tag = 'debug_' if debug else ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    if suffix is not None:
        tag += '_' + suffix
    print('Tag: {:s}'.format(tag))
    return tag

def get_transformer(patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    loading_transformations = [
        A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
        A.Resize(height=patch_size, width=patch_size, always_apply=True),
        ]

    if train:
        aug_transformations = [
            A.HorizontalFlip(p=0.3), 

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.5),
                A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=5, p=0.5),
            ], p=0.4),
        ]
    else:
        aug_transformations = []

    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
        ToTensorV2()
    ]

    return A.Compose(loading_transformations + aug_transformations + final_transformations)

def aggregate(x, deadzone: float, pre_mult: float, policy: str, post_mult: float, clipmargin: float, params={}):
    x = x.copy()
    if deadzone > 0:
        x = x[(x > deadzone) | (x < -deadzone)]
        if len(x) == 0:
            x = np.asarray([0, ])
    if policy == 'mean':
        x = np.mean(x)
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'sigmean':
        x = scipy.special.expit(x * pre_mult).mean()
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'meanp':
        pow_coeff = params.pop('p', 3)
        x = np.mean(np.sign(x) * (np.abs(x) ** pow_coeff))
        x = np.sign(x) * (np.abs(x) ** (1 / pow_coeff))
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'median':
        x = scipy.special.expit(np.median(x) * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'sigmedian':
        x = np.median(scipy.special.expit(x * pre_mult))
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'maxabs':
        x = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'avgvoting':
        x = np.mean(np.sign(x))
        x = (x * post_mult + 1) / 2
    elif policy == 'voting':
        x = np.sign(np.mean(x * pre_mult))
        x = (x - 0.5) * post_mult + 0.5
    else:
        raise NotImplementedError()
    return np.clip(x, clipmargin, 1 - clipmargin)