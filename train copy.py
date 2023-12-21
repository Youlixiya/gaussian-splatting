#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, GaussianFeatureModel
from scene.camera_scene import CamScene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.extract_masks import MaskDataset
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_cosine_similarities(vectors, query_vector):
    return F.cosine_similarity(query_vector.unsqueeze(0), vectors, dim=1)

def training(args, dataset, opt, pipe, saving_iterations):
    cur_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianFeatureModel(dataset.sh_degree, rgb_decode=args.rgb_decode, depth_decode=args.depth_decode, device='cuda')
    gaussians.load_ply(args.gs_source)
    img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
    h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
    scene = CamScene(args.colmap_dir, h=h, w=w)

    feature_bg_color = torch.tensor([0] * gaussians.instance_feature_dim, dtype=torch.float32, device="cuda")
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    loss_for_log = 0.0
    downscale = '' if args.images == 'images' else args.images.split('_')[-1]
    downscale = downscale if downscale == '' else f'_{downscale}'
    mask_dataset = MaskDataset(args.colmap_dir, scene.cameras.copy(), mask_dir=f'masks{downscale}')
    print(f'instance num: {len(mask_dataset.instance_colors)}')
    gaussians.set_instance_embeddings(len(mask_dataset.instance_colors))
    gaussians.set_clip_embeddings(mask_dataset.clip_embeddings)
    gaussians.set_instance_colors(mask_dataset.instance_colors)
    gaussians.feature_training_setup(opt)
    progress_bar = tqdm(range(cur_iter, opt.feature_iterations), desc="Training Feature GS progress")
    batch_size = opt.batch_size
    temperature = opt.temperature
    while cur_iter < opt.feature_iterations:
        cur_iter += 1
        iter_start.record()
        index = randint(0, len(mask_dataset)-1)
        viewpoint_stack = scene.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(index)
        with torch.no_grad():
            rendered_gs = render(viewpoint_cam, gaussians, pipe, bg_color)
        rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, override_feature=gaussians.instance_features)['render_feature']
        total_feature = [rendered_feature]
        if args.rgb_decode:
            total_feature.append(rendered_gs['render'])
        if args.depth_decode:
            total_feature.append(rendered_gs['depth_3dgs'])
        total_feature = torch.cat(total_feature, dim=0)
        h, w = rendered_feature.shape[1:]
        instance_masks = mask_dataset.instance_masks[index]
        pixel_index = torch.randint(0, h * w, (batch_size, ))
        # instance embedding contrastive loss
        instance_embeddings = F.normalize(gaussians.instance_embeddings, dim=-1)
        # instance_embeddings = gaussians.instance_embeddings
        instance_contrastive_matrix = temperature * instance_embeddings @ instance_embeddings.T.detach()
        gt_instance_contrastive_labels = torch.arange(gaussians.instance_embeddings.shape[0], device='cuda')
        instance_contrastive_loss = F.cross_entropy(instance_contrastive_matrix, gt_instance_contrastive_labels)
        # rendered embedding contrastive loss
        # instance_masks_batch = instance_masks.reshape(h * w)[pixel_index]
        # total_feature_batch = F.normalize(total_feature.reshape(-1, h * w).permute(1, 0)[pixel_index])
        # instance_embeddings_index = torch.argmax(total_feature_batch @ instance_embeddings.T, dim=-1)
        
        # total_feature_contrastive_matrix = 
        # instance loss
        instance_masks_batch = instance_masks.reshape(h * w)[pixel_index]
        total_feature_batch = F.normalize(total_feature.reshape(-1, h * w).permute(1, 0)[pixel_index], dim=-1)
        # total_feature_batch = total_feature.reshape(-1, h * w).permute(1, 0)[pixel_index]
        pred_instance_matrix = total_feature_batch @ instance_embeddings.T.detach()
        # target_similarity = pred_instance_matrix[:, instance_masks_batch]
        # target_similarity = 
        instance_loss = F.cross_entropy(pred_instance_matrix, instance_masks_batch)

        # norm loss
        # norm_loss = torch.norm(total_feature, dim=-1).mean()
        
        loss = opt.contrastive_ratio * instance_contrastive_loss + instance_loss
        loss.backward()

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            loss_for_log = loss.item()
            # loss_for_log = total_loss.item()
            # if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss_for_log:.{7}f}"})
            progress_bar.update(1)
            if cur_iter + 1 == opt.feature_iterations:
                progress_bar.close()
            if (cur_iter + 1 in saving_iterations):
                print("\n[ITER {}] Saving Feature Gaussians".format(cur_iter + 1))
                save_path = os.path.abspath(os.path.join(args.gs_source, os.pardir))
                gaussians.save_feature_params(save_path, cur_iter + 1)

            # Optimizer step
            if cur_iter + 1 < opt.feature_iterations:
                gaussians.optimizer.step()
                # scaler.step(gaussians.optimizer)
                gaussians.optimizer.zero_grad(set_to_none = True)
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    parser.add_argument("--rgb_decode", action='store_true')
    parser.add_argument("--depth_decode", action='store_true')
    args = parser.parse_args(sys.argv[1:])
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations)

    # All done
    print("\nTraining complete.")
