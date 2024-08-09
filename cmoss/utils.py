import glob
import os
from gzip import GzipFile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from lve.utils import torch_float_01_to_np_uint8
import random


class SimpleMetric():
    def __init__(self, l):
        self.val = 0.0
        self.l = l

    def accumulate(self, v):
        self.val += v

    def get(self, d=None):
        return self.val / (self.l if d is None else d)

    def reset(self):
        self.val = 0.0


def parse_augmentation(args_cmd):
    colorjitter_prob = args_cmd.augmentation['colorjitter'][0]
    colorjitter_intensity = args_cmd.augmentation['colorjitter'][1]
    lower_bound = 1 - colorjitter_intensity
    upper_bound = 1 + colorjitter_intensity
    return {
        'brightness': (lower_bound, upper_bound),
        'contrast': (lower_bound, upper_bound),
        'hue': (- colorjitter_intensity, colorjitter_intensity),
        'saturation': (lower_bound, upper_bound),
        'prob': colorjitter_prob
    }


# def sample_jitter_params(brightness=(1.0, 1.0), contrast=(1.0, 1.0), saturation=(1.0, 1.0), hue=(0.0, 0.0)):
def sample_jitter_params(brightness=(0.7, 1.5), contrast=(0.8, 1.2), saturation=(0.4, 1.6), hue=(-0.3, 0.3), gaussian=(.1, 2)):
    b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
    c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
    s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
    h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
    g = None if gaussian is None else float(torch.empty(1).uniform_(gaussian[0], gaussian[1]))
    return b, c, s, h, g


def jitter_frames(x, b, c, s, h, g):
    x = F.adjust_brightness(x, b)
    x = F.adjust_contrast(x, c)
    x = F.adjust_saturation(x, s)
    x = F.adjust_hue(x, h)
    x = F.gaussian_blur(x, 7, g)
    return x


class PairsOfFramesDataset(Dataset):

    def __init__(self, root_dir, device, force_gray=True, fix_motion_u=False, fix_motion_v=False, n=1,
                 input_normalization="no", motion_disk_type=None, augmentation=False, bgr_input=True):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(root_dir + os.sep + "frames" + os.sep + "**" + os.sep + "*.png", recursive=True))
        self.motion_disk_type = motion_disk_type
        if motion_disk_type is not None:
            self.motion_disk_type_files = sorted(
                glob.glob(root_dir + os.sep + motion_disk_type + os.sep + "**" + os.sep + "*.bin", recursive=True))

        self.motion_files = sorted(
            glob.glob(root_dir + os.sep + "motion" + os.sep + "**" + os.sep + "*.bin", recursive=True))
        self.motion_available = len(self.motion_files) > 0
        self.force_gray = force_gray
        self.length = len(self.files) - n  # remove last frame
        self.device = device
        self.fix_motion_u = fix_motion_u
        self.fix_motion_v = fix_motion_v
        self.n = n
        self.input_normalization = input_normalization == "yes"
        self.augmentation = augmentation
        self.bgr_input = bgr_input

    def __len__(self):
        return self.length

    def transform(self, old_frame, frame, motion, motion_flag=False):
        cropped_frame = frame
        cropped_old_frame = old_frame
        # Random resized crop
        if random.random() < self.augmentation['crop']:
            crop = transforms.RandomResizedCrop(size=old_frame.shape[1:])  # check if 256
            params = crop.get_params(old_frame, scale=(0.35, 1.0), ratio=(0.85, 1.1))

            cropped_old_frame = transforms.functional.resized_crop(old_frame, *params, size=old_frame.shape[1:])
            cropped_frame = transforms.functional.resized_crop(frame, *params, size=frame.shape[1:])
            if motion_flag:
                i, j, h, w = params
                h_ratio = h / old_frame.shape[1]  # ratio new_height/old_height
                w_ratio = w / old_frame.shape[2]
                motion = transforms.functional.resized_crop(motion, *params, size=motion.shape[1:])
                motion[0] = motion[0] / h_ratio
                motion[1] = motion[1] / w_ratio

        # cropped_old_frame = old_frame
        # cropped_frame = frame
        # Random horizontal flipping
        if self.augmentation['flip'] and random.random() < self.augmentation['flip']:
            cropped_old_frame = F.hflip(cropped_old_frame)
            cropped_frame = F.hflip(cropped_frame)
            if motion_flag:
                motion = F.hflip(motion)
                motion[0] = - motion[0]

        # Random vertical flipping
        if self.augmentation['flip'] and random.random() < self.augmentation['flip']:
            cropped_old_frame = F.vflip(cropped_old_frame)
            cropped_frame = F.vflip(cropped_frame)
            if motion_flag:
                motion = F.vflip(motion)
                motion[1] = - motion[1]

        if self.augmentation['colordropout'] > 0:
            dropout_mask = (torch.FloatTensor(3, 1, 1).uniform_().to(cropped_frame.device) > self.augmentation[
                'colordropout']).float()
            cropped_old_frame *= dropout_mask
            cropped_frame *= dropout_mask

        if self.augmentation['colorjitter']['prob'] > 0. and random.random() < self.augmentation['colorjitter']['prob']:
            b, c, s, h = sample_jitter_params(brightness=self.augmentation['colorjitter']['brightness'],
                                              contrast=self.augmentation['colorjitter']['contrast'],
                                              hue=self.augmentation['colorjitter']['hue'],
                                              saturation=self.augmentation['colorjitter']['saturation'])
            cropped_frame = jitter_frames(cropped_frame, b, c, s, h)
            cropped_old_frame = jitter_frames(cropped_old_frame, b, c, s, h)

        return cropped_old_frame, cropped_frame, motion

    def __getitem__(self, idx):
        old_frame = cv2.imread(self.files[idx])
        frame = cv2.imread(self.files[idx + self.n])
        if not self.bgr_input:
            # convert to rgb only ion case of False flag - notice that this is different from what done before
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)

        if self.force_gray and frame.shape[2] > 1:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (frame.shape[0], frame.shape[1], 1))

        if self.motion_available:
            with GzipFile(self.motion_files[idx + self.n]) as f:
                motion = np.load(f)
                motion = torch.from_numpy(motion.transpose(2, 0, 1)).float()
            if self.fix_motion_v: motion[1] *= -1
            if self.fix_motion_u: motion[0] *= -1
        else:
            motion = torch.empty(1)

        if self.force_gray and old_frame.shape[2] > 1:
            old_frame = np.reshape(cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY),
                                   (old_frame.shape[0], old_frame.shape[1], 1))

        if self.motion_disk_type is not None:
            with GzipFile(self.motion_disk_type_files[idx]) as f:
                motion_disk_files = np.load(f)
                motion_disk_files = torch.from_numpy(motion_disk_files).float()
        else:
            motion_disk_files = torch.empty(1)

        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div_(255.0)
        old_frame = torch.from_numpy(old_frame.transpose(2, 0, 1)).float().div_(255.0)
        if self.input_normalization:
            frame = (frame - 0.5) / 0.25
            old_frame = (old_frame - 0.5) / 0.25

        if self.augmentation:
            old_frame, frame, motion_disk_files = self.transform(old_frame, frame, motion_disk_files,
                                                                 self.motion_disk_type is not None)

        return (old_frame, frame, motion, motion_disk_files, idx)


def compute_farneback_motion(old_frame, frame, backward=False):
    if backward:
        frames = (frame, old_frame)
    else:
        frames = (old_frame, frame)
    return cv2.calcOpticalFlowFarneback(frames[0],
                                        frames[1],
                                        None,
                                        pyr_scale=0.4,
                                        levels=5,  # pyramid levels
                                        winsize=12,
                                        iterations=10,
                                        poly_n=5,
                                        poly_sigma=1.1,
                                        flags=0)


def compute_motions(frames, old_frames, backward=False):
    frames_numpy = torch_float_01_to_np_uint8(frames)
    old_frames_numpy = torch_float_01_to_np_uint8(old_frames)
    h, w = frames_numpy.shape[1:3]
    motions = []
    for i in range(frames_numpy.shape[0]):
        frame_gray = cv2.cvtColor(frames_numpy[i], cv2.COLOR_BGR2GRAY).reshape(h, w, 1)
        old_frame_gray = cv2.cvtColor(old_frames_numpy[i], cv2.COLOR_BGR2GRAY).reshape(h, w, 1)
        motions.append(compute_farneback_motion(frame=frame_gray, old_frame=old_frame_gray, backward=backward))
    return torch.tensor(np.array(motions)).permute(0, 3, 1, 2)


## from pwc code

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def get_occu_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def get_occu_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()
