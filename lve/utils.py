import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import flow_vis
import wandb
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import ceil
import torch.nn.functional as F

NORMALIZATION_FACTORS_FOR_DATASETS = {
    'toy_bench': 5.0,
    'empty_space_bench': 3.0,
    'empty_space_bench_small': 3.0,
    'small_test': 3.0,
    'solid_benchmark': 3.0,
    'short_stream_a': 3.0,
    'rat_small': 8.5,
    'rat_small_eval': 8.5,
    'horse_small': 8.5,
    'horse_small_eval': 8.5,
}

device_cpu = torch.device("cpu")


def load_json(json_file_path):
    f = open(json_file_path, "r")
    if f is None or not f or f.closed:
        raise IOError("Cannot read: " + json_file_path)
    json_loaded = json.load(f)
    f.close()
    return json_loaded


def save_json(json_file_path, json_to_save):
    f = open(json_file_path, "w")
    if f is None or not f or f.closed:
        raise IOError("Cannot access: " + json_file_path)
    json.dump(json_to_save, f, indent=4)
    f.close()


def np_uint8_to_torch_float_01(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).float().div_(255.0).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).float().resize_(1, 1, h, w).to(device).div_(255.0)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().to(device).unsqueeze_(0).div_(255.0)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().div_(255.0)
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device).div_(255.0)
    else:
        raise ValueError("Unsupported image type.")


def np_float32_to_torch_float(numpy_img, device=None):
    if numpy_img.ndim == 2:
        h = numpy_img.shape[0]
        w = numpy_img.shape[1]
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w)
        else:
            return torch.from_numpy(numpy_img).resize_(1, 1, h, w).to(device)
    elif numpy_img.ndim == 3:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0)
        else:
            return torch.from_numpy(numpy_img.transpose(2, 0, 1)).float().unsqueeze_(0).to(device)
    elif numpy_img.ndim == 4:
        if device is None or device == device_cpu:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float()
        else:
            return torch.from_numpy(numpy_img.transpose(0, 3, 1, 2)).float().to(device)
    else:
        raise ValueError("Unsupported image type.")


def torch_float32_to_grayscale_float32(torch_img):
    return torch.sum(torch_img *
                     torch.tensor([[[[0.114]], [[0.587]], [[0.299]]]],
                                  dtype=torch.float32, device=torch_img.device), 1, keepdim=True)


def torch_float_01_to_np_uint8(torch_img):
    if torch_img.ndim == 2:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8)
    elif torch_img.ndim == 3:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    elif torch_img.ndim == 4:
        return (torch_img * 255.0).cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
    else:
        raise ValueError("Unsupported image type.")


def round_up_to_odd(f):
    return np.int(np.ceil(f) // 2 * 2 + 1)


def torch_2d_tensor_to_csv(tensor, file):
    with open(file, 'w+') as f:
        for i in range(0, tensor.shape[0]):
            for j in range(0, tensor.shape[1]):
                f.write(str(tensor[i, j].item()))
                if j < tensor.shape[1] - 1:
                    f.write(',')
            f.write('\n')


def indices_to_rgb(indices):
    colors = np.array(
        [[189, 183, 107], [250, 235, 215], [0, 255, 255], [127, 255, 212], [240, 255, 255], [245, 245, 220],
         [255, 228, 196], [0, 0, 0], [255, 235, 205], [0, 0, 255], [138, 43, 226], [165, 42, 42], [222, 184, 135],
         [95, 158, 160], [127, 255, 0], [210, 105, 30], [255, 127, 80], [100, 149, 237], [255, 248, 220],
         [220, 20, 60], [0, 255, 255], [0, 0, 139], [0, 139, 139], [184, 134, 11], [169, 169, 169], [0, 100, 0],
         [169, 169, 169], [240, 248, 255], [139, 0, 139], [85, 107, 47], [255, 140, 0], [153, 50, 204], [139, 0, 0],
         [233, 150, 122], [143, 188, 143], [72, 61, 139], [47, 79, 79], [47, 79, 79], [0, 206, 209], [148, 0, 211],
         [255, 20, 147], [0, 191, 255], [105, 105, 105], [105, 105, 105], [30, 144, 255], [178, 34, 34],
         [255, 250, 240], [34, 139, 34], [255, 0, 255], [220, 220, 220], [248, 248, 255], [255, 215, 0],
         [218, 165, 32], [128, 128, 128], [0, 128, 0], [173, 255, 47], [128, 128, 128], [240, 255, 240],
         [255, 105, 180], [205, 92, 92], [75, 0, 130], [255, 255, 240], [240, 230, 140], [230, 230, 250],
         [255, 240, 245], [124, 252, 0], [255, 250, 205], [173, 216, 230], [240, 128, 128], [224, 255, 255],
         [250, 250, 210], [211, 211, 211], [144, 238, 144], [211, 211, 211], [255, 182, 193], [255, 160, 122],
         [32, 178, 170], [135, 206, 250], [119, 136, 153], [119, 136, 153], [176, 196, 222], [255, 255, 224],
         [0, 255, 0], [50, 205, 50], [250, 240, 230], [255, 0, 255], [128, 0, 0], [102, 205, 170], [0, 0, 205],
         [186, 85, 211], [147, 112, 219], [60, 179, 113], [123, 104, 238], [0, 250, 154], [72, 209, 204],
         [199, 21, 133], [25, 25, 112], [245, 255, 250], [255, 228, 225], [255, 228, 181], [255, 222, 173],
         [0, 0, 128], [253, 245, 230], [128, 128, 0], [107, 142, 35], [255, 165, 0], [255, 69, 0], [218, 112, 214],
         [238, 232, 170], [152, 251, 152], [175, 238, 238], [219, 112, 147], [255, 239, 213], [255, 218, 185],
         [205, 133, 63], [255, 192, 203], [221, 160, 221], [176, 224, 230], [128, 0, 128], [255, 0, 0],
         [188, 143, 143], [65, 105, 225], [139, 69, 19], [250, 128, 114], [244, 164, 96], [46, 139, 87],
         [255, 245, 238], [160, 82, 45], [192, 192, 192], [135, 206, 235], [106, 90, 205], [112, 128, 144],
         [112, 128, 144], [255, 250, 250], [0, 255, 127], [70, 130, 180], [210, 180, 140], [0, 128, 128],
         [216, 191, 216], [255, 99, 71], [64, 224, 208], [238, 130, 238], [245, 222, 179], [255, 255, 255],
         [245, 245, 245], [255, 255, 0], [154, 205, 50]], dtype=np.uint8)
    return colors[indices.astype(np.int64)]


def plot_what_heatmap(what_img, what_ref, anchor=None, distance='euclidean', out='spatial_coherence_heatmap.png'):
    what_ref = what_ref.reshape(what_img.shape[1], 1, 1)
    if distance == 'euclidean':
        diff = what_img[0, :, :, :] - what_ref
        dist = np.linalg.norm(diff, axis=0)
    else:
        dist = 1.0 - np.sum((what_img[0, :, :, :] * what_ref), axis=0)
    fig, ax = plt.subplots()
    ax.axis('off')
    # plt.imshow(recon_img, cmap=plt.cm.gray)
    if anchor is not None:
        anchorx, anchory = anchor
        plt.plot([anchory], [anchorx], marker='x')
    plt.imshow(-dist, cmap=plt.cm.Greens)  # alpha=0.95
    plt.savefig(out, bbox_inches='tight', transparent=True, pad_inches=0.0)


def plot_what_heatmap_no_save(what_img, what_ref, anchor=None, distance='euclidean'):
    what_ref = what_ref.reshape(what_img.shape[1], 1, 1)
    if distance == 'euclidean':
        diff = what_img[0, :, :, :] - what_ref
        dist = np.linalg.norm(diff, axis=0)
    else:
        dist = 1.0 - np.sum((what_img[0, :, :, :] * what_ref), axis=0)
    plt.close('all')
    fig, ax = plt.subplots()
    ax.axis('off')
    # plt.imshow(recon_img, cmap=plt.cm.gray)
    if anchor is not None:
        anchorx, anchory = anchor
        plt.plot([anchory], [anchorx], marker='x')
    plt.imshow(-dist, cmap=plt.cm.Greens)  # alpha=0.95
    plt.colorbar()
    plt.show()
    return plt


def decode_frequency_features(box_width, d_feat=None, grad_sq_feat=None, num_classes=None):
    if grad_sq_feat is None:
        feat_dx, feat_dy = d_feat
        grad_sq_feat = (feat_dx ** 2) + (feat_dy ** 2)
    n, c, h1, h2 = grad_sq_feat.shape
    super_features = box_spatial_conv(grad_sq_feat, box_width)
    winning_feature_idx = torch.argmax(super_features, dim=1, keepdim=True)  # n,1,h1,h2
    out_features = F.one_hot(winning_feature_idx.view(n, h1 * h2), num_classes=num_classes).permute(0, 2, 1).view(n, c,
                                                                                                                  h1,
                                                                                                                  h2)

    return {
        'super_features': super_features,
        'out_features': out_features,
        'winning_feature_idx': winning_feature_idx
    }


def box_spatial_conv(frame, box_size=3):
    in_channels = frame.shape[1]
    filter = torch.ones((in_channels, 1, box_size, box_size), dtype=torch.float32, device=frame.device) \
             / float(box_size ** 2)
    out = torch.conv2d(torch.nn.functional.pad(frame, (box_size // 2, box_size // 2, box_size // 2, box_size // 2),
                                               mode='reflect'),
                       filter, bias=None, padding='valid', groups=in_channels)
    return out


def sampled_similarity_dissimilarity_loss(f1, f2, disp, sampled_pairs=100,
                                          dissimilarity_threshold=0.0,
                                          similarity_threshold=0.0,
                                          moving_threshold=0.1,
                                          moving_vs_static_only=False,
                                          simdis_loss='plain',  # 'plain', 'logexp'
                                          simdis_loss_tau=1.0,  # temperature in case of simdis_loss='logexp'
                                          simdis_type='single',  # 'single', 'both', 'both_mixed', 'mixed'
                                          sampling_type='plain',  # 'plain', 'motion', 'features', 'motion_features'
                                          kept_pairs_perc=1.0,
                                          sampling_features='second',
                                          simdis_neg_avg=False):
    b = f1.shape[0]
    n = f1.shape[1]
    h = f1.shape[2]
    w = f1.shape[3]
    hw = h * w

    samples = int(math.sqrt(sampled_pairs))

    assert samples >= 2
    assert samples <= hw
    assert 0.0 < kept_pairs_perc <= 1.0
    assert dissimilarity_threshold <= similarity_threshold
    assert simdis_type in ['single', 'both', 'both_mixed', 'mixed']
    assert simdis_loss in ['plain', 'logexp']
    assert sampling_type in ['plain', 'motion', 'features', 'motion_features']
    assert sampling_features in ['first', 'second']
    zero_2d = torch.zeros((1, 1), device=f1.device)

    ##################################### HACK visualization ######################################################
    ###############################################################################################################
    """
    first_el = disp[0]
    all_moving_mask = torch.linalg.norm(first_el, dim=0) > moving_threshold
    plt.imshow(all_moving_mask.detach().cpu().numpy())

    # normalize_first_el for later processing!

    first_el = torch.nn.functional.normalize(first_el, p=2.0, dim=0, eps=1e-1)

    one_index = all_moving_mask.nonzero()
    if one_index.shape[0] > 0:
        nonzero_points = one_index.shape[0]
        idx = one_index[nonzero_points//2].cpu().detach().numpy()  # get one in the middle
        plt.plot(idx[1], idx[0], 'r+')
        plt.show()

        # now plot the points similar to this one according to the threshold
        displacement_in_this_idx = first_el[:, idx[0], idx[1],  None]
        # flatten displacements
        first_el_flat = first_el.view(2, -1)

        disp_similarity = torch.matmul(displacement_in_this_idx.T,
                                       first_el_flat)
        remap_in_img_size = disp_similarity.view(h, w).cpu().numpy()

        plt.imshow(remap_in_img_size)
        plt.title("Displacement similarity w.r.t the point (+)")
        plt.colorbar()
        plt.plot(idx[1], idx[0], 'r+')
        plt.show()
        exit()
    """
    ###############################################################################################################
    ###############################################################################################################

    motion_norm = torch.linalg.norm(disp, dim=1, keepdim=True)
    norm_average = torch.mean(motion_norm, dim=[2, 3])  # b x 1
    moving_threshold = torch.clamp(norm_average, min=moving_threshold)  # max(norm, thresh) - b x 1
    moving_hw = (motion_norm >= moving_threshold.view(-1, 1, 1, 1)).to(torch.float)  # b x 1 x h x w
    num_moving = torch.sum(moving_hw, dim=(2, 3), keepdim=True)

    # sampling some points (same probability of sampling points on moving and not-moving areas)
    if sampling_type == 'plain':
        idxs1 = torch.floor(torch.rand((b, 1, samples),
                                       device=f1.device) * hw).to(torch.long)  # b x 1 x samples, each in [0, hw)
    elif sampling_type == 'motion':
        static_hw = 1. - moving_hw
        num_static = hw - num_moving

        prob_dist = moving_hw / (num_moving + 1e-10) + static_hw / (num_static + 1e-10)

        idxs1 = torch.multinomial(prob_dist.view(b, hw),
                                  samples,
                                  replacement=True).view(b, 1, samples)  # b x 1 x samples, each in [0, hw)
    elif sampling_type == 'features':

        f_s = f1 if sampling_features == "first" else f2
        winning = torch.argmax(torch.abs(f_s), dim=1).view(b, hw)  # b x hw
        winning = torch.nn.functional.one_hot(winning, num_classes=n)  # b x hw x n
        prob_dist = torch.sum(winning / (torch.sum(winning, dim=1, keepdim=True) + 1e-10), dim=2)  # b x hw

        idxs1 = torch.multinomial(prob_dist,
                                  samples,
                                  replacement=True).view(b, 1, samples)  # b x 1 x samples, each in [0, hw)
    elif sampling_type == 'motion_features':
        f_s = f1 if sampling_features == "first" else f2
        winning_feat_idx = torch.argmax(torch.abs(f_s), dim=1).view(b, hw)  # b x hw
        winning_feat_1_hot = torch.nn.functional.one_hot(winning_feat_idx, num_classes=n)  # b x hw x n

        static_hw = 1. - moving_hw  # b x 1 x h x w
        winning_static = winning_feat_1_hot * static_hw.view(b, hw, 1).expand(b, hw, n)  # zeroing moving pixels
        winning_moving = winning_feat_1_hot * moving_hw.view(b, hw, 1).expand(b, hw, n)  # zeroing static pixels

        prob_dist = \
            torch.sum(winning_static / (torch.sum(winning_static, dim=1, keepdim=True) + 1e-10), dim=2) + \
            torch.sum(winning_moving / (torch.sum(winning_moving, dim=1, keepdim=True) + 1e-10), dim=2)  # b x hw

        idxs1 = torch.multinomial(prob_dist,
                                  samples,
                                  replacement=True).view(b, 1, samples)  # b x 1 x samples, each in [0, hw)
    """
    for i in range(0, b):
        P_sampled = torch.zeros((h, w))
        P_sampled.view(-1)[idxs1[i, 0, :]] = 1.
        plt.imshow(P_sampled.numpy(), cmap="hot")
        plt.waitforbuttonpress()
    """

    # motion of the sampled points
    disp = torch.gather(disp.view(b, 2, hw), 2, idxs1.expand(b, 2, samples))  # b x 2 x samples

    # coordinates of the sampled points
    ii1 = torch.div(idxs1, w, rounding_mode='trunc')
    jj1 = idxs1 % h
    ii_jj1 = torch.cat((ii1.to(torch.float).view(b, samples, 1),
                        jj1.to(torch.float).view(b, samples, 1)), dim=2)  # b x samples x 2

    # idx that move
    moving = torch.linalg.norm(disp, dim=1, keepdim=True) >= moving_threshold.view(-1, 1, 1)  # b x 1 x samples

    # pairs where both the points move
    moving_pairs = torch.logical_and(moving.transpose(1, 2), moving)  # b x samples x samples

    # two different types of masks
    if not moving_vs_static_only:

        # idx that do not move
        # not_moving = torch.logical_not(moving)  # b x 1 x samples

        # # idx where both do not move
        # not_moving_pairs = torch.logical_and(not_moving.transpose(1, 2), not_moving)  # b x samples x samples

        # normalized motion
        disp = torch.nn.functional.normalize(disp, p=2.0, dim=1, eps=1e-1)  # b x 2 x samples

        # motion similarity (pairs)
        disp_similarity = torch.matmul(disp.transpose(1, 2),
                                       disp)  # matmul(b x samples x 2, b x 2 x samples) = b x samples x samples

        # pairs with not similar motion AND where both the points are moving or only one of them is moving
        # dis_mask = torch.logical_and(disp_similarity <= dissimilarity_threshold,
        #                              torch.logical_not(not_moving_pairs)).to(torch.float)  # b x samples x samples

        dis_mask = torch.logical_and(disp_similarity <= dissimilarity_threshold, moving_pairs).to(torch.float)

        # batched distances between pairs of points
        dists = torch.cdist(ii_jj1, ii_jj1, p=2.0)  # b x samples x samples

        # the farther the better
        dis_mask_dists = dists * dis_mask
        dis_mask_dists = dis_mask_dists / (torch.max(dis_mask_dists) + 1e-10)

        one_moving_pairs = torch.logical_xor(moving.transpose(1, 2), moving)
        dis_mask = dis_mask_dists + one_moving_pairs.to(torch.float)

        # pairs with similar motion AND where both the points are moving
        sim_mask = torch.logical_and(disp_similarity > similarity_threshold,
                                     moving_pairs).to(torch.float)  # b x samples x samples

        # the closer the better
        sim_mask = sim_mask * (1. - (dists / (torch.max(dists) + 1e-10)))
    else:

        # pairs where only one point is moving
        one_moving_pairs = torch.logical_xor(moving.transpose(1, 2), moving)  # take pairs with only one moving
        dis_mask = one_moving_pairs.to(torch.float)
        sim_mask = moving_pairs.to(torch.float)

    # init elements that will be eventually/partially overwritten by the following code
    sim_loss_not_aggregated1 = zero_2d
    dis_loss_not_aggregated1 = zero_2d
    sim_loss_not_aggregated2 = zero_2d
    dis_loss_not_aggregated2 = zero_2d
    sim_loss_not_aggregated12 = zero_2d
    dis_loss_not_aggregated12 = zero_2d
    loss_not_aggregated1 = zero_2d
    loss_not_aggregated2 = zero_2d
    loss_not_aggregated12 = zero_2d
    num_sim_pairs1 = 0.
    num_dis_pairs1 = 0.
    num_sim_pairs2 = 0.
    num_dis_pairs2 = 0.
    num_sim_pairs12 = 0.
    num_dis_pairs12 = 0.
    alpha1 = 0.
    alpha2 = 0.
    alpha12 = 0.
    loss_sim = 0.
    loss_dis = 0.
    dis_idx1 = None
    dis_idx2 = None
    dis_idx12 = None

    # features of the sampled points
    f1 = torch.gather(f1.view(b, n, hw), 2, idxs1.expand(b, n, samples))  # b x n x samples
    f1_normalized = torch.nn.functional.normalize(f1, p=2.0, dim=1, eps=1e-10)  # b x 2 x samples

    if simdis_type == 'single' or simdis_type == 'both' or simdis_type == 'both_mixed':
        # dot products
        dot1 = torch.matmul(f1_normalized.transpose(1, 2), f1_normalized)

        # dot products of pairs of features in [0, 1]: we want them orthogonal => the product must be low
        dis_loss_not_aggregated1 = (dot1 + 1) / 2.

        # negative dot products of pairs of features in [0, 1], where 0 means max similarity
        # we want them parallel => the product must be high, so this term must be low (minimum in zero, max in 1)
        sim_loss_not_aggregated1 = (2. - (dot1 + 1.)) / 2.

    # corresponding points in the next frame and associated features and losses
    if simdis_type == 'both' or simdis_type == 'both_mixed' or simdis_type == 'mixed':
        ii_jj2 = torch.round(ii_jj1 + disp.transpose(1, 2)).to(torch.long)
        ii_jj2[:, :, 0] = torch.clamp(ii_jj2[:, :, 0], 0, h - 1)
        ii_jj2[:, :, 1] = torch.clamp(ii_jj2[:, :, 1], 0, w - 1)
        idxs2 = (ii_jj2[:, :, 0] * w + ii_jj2[:, :, 1]).view(b, 1, samples)

        f2 = torch.gather(f2.view(b, n, hw), 2, idxs2.expand(b, n, samples))  # b x n x samples
        f2_normalized = torch.nn.functional.normalize(f2, p=2.0, dim=1, eps=1e-10)  # b x 2 x samples

        if simdis_type == 'both' or simdis_type == 'both_mixed':
            dot2 = torch.matmul(f2_normalized.transpose(1, 2), f2_normalized)
            dis_loss_not_aggregated2 = (dot2 + 1) / 2.
            sim_loss_not_aggregated2 = (2. - (dot2 + 1.)) / 2.

        if simdis_type == 'both_mixed' or simdis_type == 'mixed':
            dot12 = torch.matmul(f1_normalized.transpose(1, 2), f2_normalized)
            dis_loss_not_aggregated12 = (dot12 + 1) / 2.
            sim_loss_not_aggregated12 = (2. - (dot12 + 1.)) / 2.

    # alternatives:
    # sum_squared_diffs = torch.exp(-0.01 * torch.sum(torch.pow(f.unsqueeze(-1) - f.unsqueeze(-2), 2), dim=1))
    # sum_squared_diffs = 1. / (torch.sum(torch.pow(f.unsqueeze(-1) - f.unsqueeze(-2), 2), dim=1) + 1e-10)
    # sum_squared_diffs = -torch.sum(torch.pow(f.unsqueeze(-1) - f.unsqueeze(-2), 2), dim=1)

    # printable_mask = torch.zeros((b, samples, hw), device=f.device, dtype=torch.float32)
    # for j in range(0, b):
    #     for k in range(0, samples):
    #         printable_mask[j, k, idxs[j, 0, :]] = mask[j, k, :]
    #         printable_mask[j, k, idxs[j, 0, k]] = -1
    # printable_mask = printable_mask.view(b, samples, h, w)

    if kept_pairs_perc < 1.0:

        # checking how many valid similarity pairs we have (for each image of the batch; then batch averaged)
        valid_pairs = torch.count_nonzero(sim_mask.view(b, -1), dim=1)  # b
        valid_pairs = torch.mean(valid_pairs.to(torch.float))  # over the batch

        # keeping only 'kept_pairs_perc' * 100.0% of the valid similarity pairs
        kept_pairs = max(int(valid_pairs.item() * kept_pairs_perc), 1)

        if simdis_type == 'single' or simdis_type == 'both' or simdis_type == 'both_mixed':
            masked_sim_loss_not_aggregated1 = sim_mask * sim_loss_not_aggregated1
            sim_loss_not_aggregated1, sim_idx1 = torch.topk(masked_sim_loss_not_aggregated1.view(b, -1),
                                                            kept_pairs, dim=1)
            num_sim_pairs1 = torch.sum(sim_mask.view(b, -1).gather(1, sim_idx1), dim=1)

        if simdis_type == 'both' or simdis_type == 'both_mixed':
            masked_sim_loss_not_aggregated2 = sim_mask * sim_loss_not_aggregated2
            sim_loss_not_aggregated2, sim_idx2 = torch.topk(masked_sim_loss_not_aggregated2.view(b, -1),
                                                            kept_pairs, dim=1)
            num_sim_pairs2 = torch.sum(sim_mask.view(b, -1).gather(1, sim_idx2), dim=1)

        if simdis_type == 'both_mixed' or simdis_type == 'mixed':
            masked_sim_loss_not_aggregated12 = sim_mask * sim_loss_not_aggregated12
            sim_loss_not_aggregated12, sim_idx12 = torch.topk(masked_sim_loss_not_aggregated12.view(b, -1),
                                                              kept_pairs, dim=1)
            num_sim_pairs12 = torch.sum(sim_mask.view(b, -1).gather(1, sim_idx12), dim=1)

        # checking how many valid dissimilarity pairs we have (for each image of the batch; then batch averaged)
        valid_pairs = torch.count_nonzero(dis_mask.view(b, -1), dim=1)  # b
        valid_pairs = torch.mean(valid_pairs.to(torch.float))  # over the batch

        # keeping only 'kept_pairs_perc' * 100.0% of the valid dissimilarity pairs
        kept_pairs = max(int(valid_pairs.item() * kept_pairs_perc), 1)

        if simdis_type == 'single' or simdis_type == 'both' or simdis_type == 'both_mixed':
            masked_dis_loss_not_aggregated1 = dis_mask * dis_loss_not_aggregated1
            dis_loss_not_aggregated1, dis_idx1 = torch.topk(masked_dis_loss_not_aggregated1.view(b, -1),
                                                            kept_pairs, dim=1)
            num_dis_pairs1 = torch.sum(dis_mask.view(b, -1).gather(1, dis_idx1), dim=1)

        if simdis_type == 'both' or simdis_type == 'both_mixed':
            masked_dis_loss_not_aggregated2 = dis_mask * dis_loss_not_aggregated2
            dis_loss_not_aggregated2, dis_idx2 = torch.topk(masked_dis_loss_not_aggregated2.view(b, -1),
                                                            kept_pairs, dim=1)
            num_dis_pairs2 = torch.sum(dis_mask.view(b, -1).gather(1, dis_idx2), dim=1)

        if simdis_type == 'both_mixed' or simdis_type == 'mixed':
            masked_dis_loss_not_aggregated12 = dis_mask * dis_loss_not_aggregated12
            dis_loss_not_aggregated12, dis_idx12 = torch.topk(masked_dis_loss_not_aggregated12.view(b, -1),
                                                              kept_pairs, dim=1)
            num_dis_pairs12 = torch.sum(dis_mask.view(b, -1).gather(1, dis_idx12), dim=1)

    else:
        if simdis_type == 'single' or simdis_type == 'both' or simdis_type == 'both_mixed':
            sim_loss_not_aggregated1 = (sim_mask * sim_loss_not_aggregated1).view(b, -1)
            dis_loss_not_aggregated1 = (dis_mask * dis_loss_not_aggregated1).view(b, -1)
            num_sim_pairs1 = torch.sum(sim_mask.view(b, -1), dim=1)
            num_dis_pairs1 = torch.sum(dis_mask.view(b, -1), dim=1)

        if simdis_type == 'both' or simdis_type == 'both_mixed':
            sim_loss_not_aggregated2 = (sim_mask * sim_loss_not_aggregated2).view(b, -1)
            dis_loss_not_aggregated2 = (dis_mask * dis_loss_not_aggregated2).view(b, -1)
            num_sim_pairs2 = num_sim_pairs1
            num_dis_pairs2 = num_dis_pairs1

        if simdis_type == 'both_mixed' or simdis_type == 'mixed':
            sim_loss_not_aggregated12 = (sim_mask * sim_loss_not_aggregated12).view(b, -1)
            dis_loss_not_aggregated12 = (dis_mask * dis_loss_not_aggregated12).view(b, -1)
            num_sim_pairs12 = num_sim_pairs1
            num_dis_pairs12 = num_dis_pairs1

    # aggregating the measures for each image of the batch
    if simdis_type == 'single':
        alpha1 = 1.0
        alpha2 = 0.0
        alpha12 = 0.0
    elif simdis_type == 'both':
        alpha1 = 0.5
        alpha2 = 0.5
        alpha12 = 0.0
    elif simdis_type == 'both_mixed':
        alpha1 = 0.333333
        alpha2 = 0.333333
        alpha12 = 0.333333
    elif simdis_type == 'mixed':
        alpha1 = 0.0
        alpha2 = 0.0
        alpha12 = 1.0

    if simdis_loss == 'plain':
        loss_sim = \
            alpha1 * torch.sum(sim_loss_not_aggregated1, dim=1) / (num_sim_pairs1 + 1e-10) + \
            alpha2 * torch.sum(sim_loss_not_aggregated2, dim=1) / (num_sim_pairs2 + 1e-10) + \
            alpha12 * torch.sum(sim_loss_not_aggregated12, dim=1) / (num_sim_pairs12 + 1e-10)  # b
        loss_dis = \
            alpha1 * torch.sum(dis_loss_not_aggregated1, dim=1) / (num_dis_pairs1 + 1e-10) + \
            alpha2 * torch.sum(dis_loss_not_aggregated2, dim=1) / (num_dis_pairs2 + 1e-10) + \
            alpha12 * torch.sum(dis_loss_not_aggregated12, dim=1) / (num_dis_pairs12 + 1e-10)  # b
    elif simdis_loss == 'logexp':
        tau = simdis_loss_tau
        zeros = torch.zeros_like(sim_mask).view(b, -1)

        if simdis_type == 'single' or simdis_type == 'both' or simdis_type == 'both_mixed':
            if kept_pairs_perc < 1.0:
                dis_mask1 = torch.scatter(zeros, 1, dis_idx1, 1.).view(b, samples, samples) * dis_mask
                sim_mask1 = torch.scatter(zeros, 1, sim_idx1, 1.).view(b, samples, samples) * sim_mask
            else:
                dis_mask1 = dis_mask
                sim_mask1 = sim_mask
            dot1 = dot1 / tau
            exp_dot = torch.exp(dot1)
            num_dis_pairs1 = num_dis_pairs1.view(b, 1, 1)
            if simdis_neg_avg:
                loss_not_aggregated1 = -dot1 + torch.log(torch.sum(dis_mask1 * exp_dot, dim=2, keepdim=True) / (num_dis_pairs1 + 1e-10)
                                                         + exp_dot + 1e-20)
            else:
                loss_not_aggregated1 = -dot1 + torch.log(
                    torch.sum(dis_mask1 * exp_dot, dim=2, keepdim=True) + exp_dot + 1e-20)
            loss_not_aggregated1 = (sim_mask1 * loss_not_aggregated1).view(b, -1)

        if simdis_type == 'both' or simdis_type == 'both_mixed':
            if kept_pairs_perc < 1.0:
                dis_mask2 = torch.scatter(zeros, 1, dis_idx2, 1.).view(b, samples, samples) * dis_mask
                sim_mask2 = torch.scatter(zeros, 1, sim_idx2, 1.).view(b, samples, samples) * sim_mask
            else:
                dis_mask2 = dis_mask
                sim_mask2 = sim_mask
            dot2 = dot2 / tau
            exp_dot = torch.exp(dot2)
            num_dis_pairs2 = num_dis_pairs2.view(b, 1, 1)
            if simdis_neg_avg:
                loss_not_aggregated2 = -dot2 + torch.log(torch.sum(dis_mask2 * exp_dot, dim=2, keepdim=True) / (num_dis_pairs2 + 1e-10)
                                                         + exp_dot + 1e-20)
            else:
                loss_not_aggregated2 = -dot2 + torch.log(
                    torch.sum(dis_mask2 * exp_dot, dim=2, keepdim=True) + exp_dot + 1e-20)
            loss_not_aggregated2 = (sim_mask2 * loss_not_aggregated2).view(b, -1)

        if simdis_type == 'both_mixed' or simdis_type == 'mixed':
            if kept_pairs_perc < 1.0:
                dis_mask12 = torch.scatter(zeros, 1, dis_idx12, 1.).view(b, samples, samples) * dis_mask
                sim_mask12 = torch.scatter(zeros, 1, sim_idx12, 1.).view(b, samples, samples) * sim_mask
            else:
                dis_mask12 = dis_mask
                sim_mask12 = sim_mask
            dot12 = dot12 / tau
            exp_dot = torch.exp(dot12)
            num_dis_pairs12 = num_dis_pairs12.view(b, 1, 1)
            if simdis_neg_avg:
                loss_not_aggregated12 = -dot12 + torch.log(torch.sum(dis_mask12 * exp_dot, dim=2, keepdim=True) / (num_dis_pairs12 + 1e-10)
                                                           + exp_dot + 1e-20)
            else:
                loss_not_aggregated12 = -dot12 + torch.log(
                    torch.sum(dis_mask12 * exp_dot, dim=2, keepdim=True) + exp_dot + 1e-20)
            loss_not_aggregated12 = (sim_mask12 * loss_not_aggregated12).view(b, -1)

        loss_sim = \
            alpha1 * torch.sum(loss_not_aggregated1, dim=1) / (num_sim_pairs1 + 1e-10) + \
            alpha2 * torch.sum(loss_not_aggregated2, dim=1) / (num_sim_pairs2 + 1e-10) + \
            alpha12 * torch.sum(loss_not_aggregated12, dim=1) / (num_sim_pairs12 + 1e-10)
        loss_dis = torch.zeros_like(loss_sim)

    # averaging over the batch (discarding static cases)
    moving_frames = (num_moving.view(b) > 0).to(torch.float)  # b
    loss_sim = torch.sum(loss_sim * moving_frames) / (torch.sum(moving_frames) + 1e-10)
    loss_dis = torch.sum(loss_dis * moving_frames) / (torch.sum(moving_frames) + 1e-10)

    # tau = 1.0
    # loss_sim = -torch.mean(torch.log(torch.sum(sim_mask * torch.exp((1. - sim_loss_not_aggregated) / tau), dim=1, keepdim=True) / \
    #           (torch.sum(sim_mask, dim=1, keepdim=True) + 1e-8) + 1e-8))  # b x 1 x samples
    # loss_dis = torch.mean(torch.log(torch.sum(dis_mask * torch.exp(dis_loss_not_aggregated / tau), dim=1, keepdim=True) / \
    #           (torch.sum(dis_mask, dim=1, keepdim=True) + 1e-8) + 1e-8))  # b x 1 x samples
    # return loss_sim, loss_dis
    # loss_sim = -torch.log(1. - torch.sum(sim_mask * sim_loss_not_aggregated) / (torch.sum(sim_mask) + 1e-8) + 1e-12)
    # loss_dis = -torch.log(1. - torch.sum(dis_mask * dis_loss_not_aggregated) / (torch.sum(dis_mask) + 1e-8) + 1e-12)

    return loss_sim, loss_dis, idxs1


def warp(old_frame, flow):
    _, _, h, w = old_frame.shape[0], old_frame.shape[1], old_frame.shape[2], old_frame.shape[3]
    warped = old_frame.clone()
    for i in range(0, h):
        for j in range(0, w):
            ii = max(min(int(round(i + flow[0][1][i][j].item())), h - 1), 0)
            jj = max(min(int(round(j + flow[0][0][i][j].item())), w - 1), 0)
            warped[0, 0, i, j] = 0.0
            warped[0, 0, ii, jj] = old_frame[0, 0, i, j]
    return warped


def backward_warp(frame, displacement):
    # _, _, h, w = old_frame.shape[0], old_frame.shape[1], old_frame.shape[2], old_frame.shape[3]
    # warped = old_frame.clone()
    # for i in range(0, h):
    #     for j in range(0, w):
    #         ii = max(min(int(round(i + flow[0][1][i][j].item())), h - 1), 0)
    #         jj = max(min(int(round(j + flow[0][0][i][j].item())), w - 1), 0)
    #         warped[0, 0, i, j] = 0.0
    #         warped[0, 0, ii, jj] = old_frame[0, 0, i, j]
    # return warped
    b, f, h, w = frame.shape
    region_h, region_w = torch.meshgrid(torch.arange(h), torch.arange(w),
                                        indexing='ij')  # NB this must be 'ij', rotation with 'xy
    region_h = region_h.to(displacement.device)
    region_w = region_w.to(displacement.device)
    u_x = 2.0 * (region_w + displacement[:, 0, :, :]) / (w - 1) - 1.0  # w-1??
    u_y = 2.0 * (region_h + displacement[:, 1, :, :]) / (h - 1) - 1.0
    ugrid = torch.stack((u_x, u_y), dim=-1)

    old_frame = torch.nn.functional.grid_sample(frame, ugrid, padding_mode='border', mode='bilinear',
                                                align_corners=True)
    return old_frame


def plot_standard_heatmap(data, underlay=None, alpha=0.3, name=None):
    if data is None:
        return None
    if not torch.is_tensor(data):
        data = torch.tensor(data)
    plt.clf()
    l = []
    n, c, h, w = data.shape
    for i in range(data.shape[0]):
        if c > 1:
            heat = torch.sqrt(torch.sum(data ** 2, dim=1))[i]
        else:
            heat = data[i, 0]

        plt.imshow(heat.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        if underlay is not None:
            plt.imshow(underlay, alpha=alpha)
        l.append(wandb.Image(plt))
        plt.clf()
    return l


def plot_sampled_points(idxs, h, w, underlay=None, alpha=0.3):
    plt.clf()
    l = []
    P_sampled = torch.zeros((h, w))
    P_sampled.view(-1)[idxs[0, 0, :]] = 1.
    plt.imshow(P_sampled.cpu().numpy(), cmap="hot")
    if underlay is not None:
        plt.imshow(underlay, alpha=alpha)
    l.append(wandb.Image(plt))
    plt.clf()
    return l


def visualize_flows(flows, prefix='flow'):
    flow_dic = {prefix + '_color': [], prefix + '_magnitude': [], prefix + '_x': [], prefix + '_y': []}
    prefix_fn = prefix.replace("/", "-")
    fnames = []
    for i in range(min(flows.shape[0], 3)):
        flow_u = flows[i, 0, :, :]
        flow_v = flows[i, 1, :, :]
        flow_uv = torch.stack((flow_u, flow_v), dim=2)
        flow_norm = torch.sqrt(torch.sum(flow_uv ** 2, dim=2))
        flow_uv = flow_uv.detach().cpu().numpy()
        flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
        flow_dic[prefix + '_color'].append(wandb.Image(flow_color))
        plt.imshow(flow_norm.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()

        r = random.randint(0, 1000000000)
        fname = prefix_fn + '_mag_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_magnitude'].append(wandb.Image(fname))

        plt.imshow(flow_u.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        fname = prefix_fn + '_x_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_x'].append(wandb.Image(fname))

        plt.imshow(flow_v.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        fname = prefix_fn + '_y_' + str(r) + '.png'
        fnames.append(fname)
        plt.savefig(fname)
        plt.clf()
        flow_dic[prefix + '_y'].append(wandb.Image(fname))
    return flow_dic, fnames


def get_normalization_factor(vproc):
    return NORMALIZATION_FACTORS_FOR_DATASETS[os.path.basename(vproc.input_stream.input_element)]


def flow_to_color_static(flow_uv, convert_to_bgr=False, static_normalization_factor=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-8
    if static_normalization_factor is not None:
        rescale = np.clip(rad / (static_normalization_factor + epsilon), 0, 1)
        u = u / (rad + epsilon) * rescale
        v = v / (rad + epsilon) * rescale
    else:
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
    return flow_vis.flow_uv_to_colors(u, v, convert_to_bgr)


def visualize_flows_no_save(flows, prefix='flow'):
    flow_dic = {prefix + '_color': [], prefix + '_magnitude': [], prefix + '_x': [], prefix + '_y': []}
    prefix_fn = prefix.replace("/", "-")
    fnames = []
    if torch.is_tensor(flows):
        flows = flows.detach().cpu().numpy()
    for i in range(min(flows.shape[0], 3)):
        flow_u = flows[i, 0, :, :]
        flow_v = flows[i, 1, :, :]
        flow_uv = np.stack((flow_u, flow_v), axis=2)
        flow_norm = np.sqrt(np.sum(flow_uv ** 2, axis=2))
        flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
        flow_dic[prefix + '_color'].append(wandb.Image(flow_color))
        plt.imshow(flow_norm, cmap='hot')
        plt.colorbar()
        plt.show()
        flow_dic[prefix + '_magnitude'].append(wandb.Image(plt))
        plt.clf()

        plt.imshow(flow_u, cmap='hot')
        plt.colorbar()
        plt.show()
        flow_dic[prefix + '_x'].append(wandb.Image(plt))
        plt.clf()

        plt.imshow(flow_v, cmap='hot')
        plt.colorbar()
        plt.show()
        flow_dic[prefix + '_y'].append(wandb.Image(plt))
        plt.clf()
    return flow_dic, fnames


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import math
import torchvision
from PIL import Image


def make_grid_show_torch(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.axis("off")  # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.tight_layout(pad=0)
    plt.axis("image")
    return plt


def make_grid_show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = (img * 255.0).astype(np.uint8)
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.axis("off")  # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.tight_layout(pad=0)
    plt.axis("image")
    return plt


def make_grid_numpy(tensor, nrow=8, padding=2, normalize=False, value_range=None,
                    scale_each=False, pad_value=0.0):
    if isinstance(tensor, list):
        tensor = np.stack(tensor, axis=0)

    if tensor.ndim == 2:
        tensor = tensor[np.newaxis, :, :]
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = np.concatenate((tensor, tensor, tensor), axis=0)
        tensor = tensor[np.newaxis, :, :, :]

    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = np.concatenate((tensor, tensor, tensor), axis=1)

    if normalize:
        tensor = tensor.copy()
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img[img < low] = low
            img[img > high] = high
            img = (img - low) / (high - low)
            print()

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each:
            for t in tensor:
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(nrow, int) or nrow <= 0:
        raise ValueError("nrow has to be a positive integer")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("padding has to be a non-negative integer")
    if not isinstance(pad_value, (float, int)):
        raise TypeError("pad_value has to be a number")

    # size of the output image
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = np.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value, dtype=np.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding: y * height + height, x * width + padding: x * width + width] = tensor[k]
            k = k + 1

    return make_grid_show(grid.transpose(1, 2, 0))


def plot_grid(images, nrow: int = 4, padding: int = 2, normalize: bool = False, colorbar=True):
    plt.close('all')
    if normalize:
        images = (images - images.min()) / (images.max() - images.min() + 1e-5)

    # Create a figure and a set of subplots

    N = images.shape[0]
    if N < 5:
        cols = N
    elif 5 <= N < 30:
        cols = 5
    else:
        cols = N // 5
    # cols = N if N < 7 else N // 4
    base_fig_size = 0.5 if N > 32 else 2.0
    fig, axs = plt.subplots(ncols=cols, nrows=ceil(N / cols),
                            figsize=(base_fig_size * cols, base_fig_size * ceil(N / cols)))
    fig.tight_layout()

    # Flatten the subplots array
    axs = axs.flatten()

    # Set the spacing between subplots to zero
    plt.subplots_adjust(wspace=.1, hspace=0.1)

    min_value, max_value = np.min(images), np.max(images)
    normalizer = Normalize(min_value, max_value)
    im = cm.ScalarMappable(norm=normalizer)

    for i, ax in enumerate(axs):
        if i < images.shape[0]:
            ax.imshow(images[i], norm=normalizer)
        ax.axis("off")

    if colorbar:
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=1.0)

    return plt


def color_plotter(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    N = max(y) + 1
    cmap = plt.cm.jet
    cmaplist = ["#81a0c2", "#f8a95d", "#ec8988", "#9cccc9"]  # colors taken from wandb
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist[:N], N)

    # define the bins and normalize

    # make the scatter
    scat = ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap)
    # create the colorbar
    plt.legend(handles=scat.legend_elements()[0], labels=range(N))
    return plt


def plot_tsne(x, y):
    tsne_states = TSNE(n_components=2, early_exaggeration=1.0, learning_rate="auto", init="pca").fit_transform(x)
    plt = color_plotter(tsne_states, y)
    return plt


def plot_pca(x, y):
    pca = PCA(n_components=2).fit_transform(x)
    plt = color_plotter(pca, y)
    return plt


def plot_umap(x, y):
    umap_red = umap.UMAP().fit_transform(x)
    plt = color_plotter(umap_red, y)
    return plt


def create_df_for_embeddings(features, targets,
                             n=None):  # features (list of np arrays), targets (list of int for each pixel)
    df = pd.DataFrame(np.array(features),
                      columns=['f' + str(i) for i in range(features.shape[1])])
    df['target'] = targets
    df["target"] = df["target"].astype(str)
    if n is not None:
        df = df.sample(n=n)
    return df


def plot_flows(flows, prefix='flow'):
    flow_dic = {prefix + '_color': [], prefix + '_magnitude': [], prefix + '_x': [], prefix + '_y': []}
    for i in range(flows.shape[0]):
        flow_u = flows[i, 0, :, :]
        flow_v = flows[i, 1, :, :]
        flow_uv = torch.stack((flow_u, flow_v), dim=2)
        flow_norm = torch.sqrt(torch.sum(flow_uv ** 2, dim=2))
        flow_uv = flow_uv.detach().cpu().numpy()
        flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
        flow_dic[prefix + '_color'].append(wandb.Image(flow_color))

        plt.clf()
        plt.imshow(flow_norm.detach().cpu().numpy(), cmap='hot')
        plt.colorbar()
        flow_dic[prefix + '_magnitude'].append(wandb.Image(plt))

        # plt.clf()
        # plt.imshow(flow_u.detach().cpu().numpy(), cmap='hot')
        # plt.colorbar()
        # flow_dic[prefix + '_x'].append(wandb.Image(plt))
        #
        # plt.clf()
        # plt.imshow(flow_v.detach().cpu().numpy(), cmap='hot')
        # plt.colorbar()
        # flow_dic[prefix + '_y'].append(wandb.Image(plt))
    return flow_dic


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx, :]


def normalize_heatmaps(video_tensor):
    m = np.min(video_tensor)
    M = np.max(video_tensor)
    video_tensor = (video_tensor - m) / (M - m)
    video_tensor = (np.clip(video_tensor, 0, 1) * 255).astype(np.uint8)
    heatmaps = np.expand_dims(video_tensor, 1)
    heatmaps = np.repeat(heatmaps, 3, axis=1)
    return heatmaps
