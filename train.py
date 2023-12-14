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
    gaussians = GaussianFeatureModel(dataset.sh_degree, device='cuda')
    gaussians.load_ply(args.gs_source)
    img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
    h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
    scene = CamScene(args.colmap_dir, h=h, w=w)

    feature_bg_color = torch.tensor([0] * gaussians.gs_feature_dim, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    loss_for_log = 0.0
    downscale = '' if args.images == 'images' else args.images.split('_')[-1]
    downscale = downscale if downscale == '' else f'_{downscale}'
    mask_dataset = MaskDataset(args.colmap_dir, scene.cameras.copy(), mask_dir=f'masks{downscale}')
    print(f'instance num: {len(mask_dataset.instance_colors)}')
    print(f'semantic num: {len(mask_dataset.semantic_colors)}')
    # gaussians.set_mask_decoder(len(mask_dataset.unique_colors))
    gaussians.set_instance_embeddings(len(mask_dataset.instance_colors))
    gaussians.set_semantic_embeddings(len(mask_dataset.semantic_colors))
    gaussians.set_clip_embeddings(mask_dataset.clip_embeddings)
    # gaussians.set_semantic_compressor(len(mask_dataset.semantic_colors))
    gaussians.set_instance_colors(mask_dataset.instance_colors, mask_dataset.semantic_colors)
    gaussians.feature_training_setup(opt)
    # fuse_semantic_table_bar = tqdm(range(len(mask_dataset)), desc="Fuse semantic table")
    progress_bar = tqdm(range(cur_iter, opt.feature_iterations), desc="Training Feature GS progress")
    batch_size = 4096
    # scaler = GradScaler()
    # temperature = 100
    while cur_iter < opt.feature_iterations:
        cur_iter += 1
        iter_start.record()
        index = randint(0, len(mask_dataset)-1)
        viewpoint_stack = scene.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(index)
        rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, override_feature=gaussians.gs_features)['render_feature']
        h, w = rendered_feature.shape[1:]
        instance_masks = mask_dataset.instance_masks[index]
        semantic_masks = mask_dataset.semantic_masks[index]
        pixel_index = torch.randint(0, h * w, (batch_size, ))
        # instance contrastive loss
        instance_contrastive_matrix = (gaussians.instance_embeddings @ gaussians.instance_embeddings.T).softmax(-1)
        gt_instance_contrastive_labels = torch.arange(gaussians.instance_embeddings.shape[0], device='cuda')
        instance_contrastive_loss = F.cross_entropy(instance_contrastive_matrix, gt_instance_contrastive_labels)
        # semantic contrastive loss
        # semantic_embedding_low_dim = gaussians.semantic_compressor(gaussians.semantic_embeddings.float())
        semantic_contrastive_matrix = (gaussians.semantic_embeddings @ gaussians.semantic_embeddings.T).softmax(-1)
        gt_semantic_contrastive_labels = torch.arange(gaussians.semantic_embeddings.shape[0], device='cuda')
        # semantic_contrastive_matrix = (semantic_embedding_low_dim @ semantic_embedding_low_dim.T).softmax(-1)
        # gt_semantic_contrastive_labels = torch.arange(semantic_embedding_low_dim.shape[0], device='cuda')
        semantic_contrastive_loss = F.cross_entropy(semantic_contrastive_matrix, gt_semantic_contrastive_labels)
        
        # render_feature_batch = render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
         #instance loss
        instance_masks_batch = instance_masks.reshape(h * w)[pixel_index]
        # instance_render_feature_batch = gaussians.instance_decoder(render_feature.reshape(-1, h * w).permute(1, 0)[pixel_index])
        instance_render_feature_batch = rendered_feature[:gaussians.instance_feature_dim].reshape(-1, h * w).permute(1, 0)[pixel_index]
        # instance_render_feature = render_feature[feature_dim:, ...]
        # instance_render_feature_batch = instance_render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
        # instance_loss = F.l1_loss(instance_render_feature_batch, gaussians.instance_embeddings[instance_masks_batch].detach())
        pred_instance_prob = (instance_render_feature_batch @ gaussians.instance_embeddings.T).softmax(-1)
        # pred_instance_prob = (gaussians.instance_decoder(render_feature_batch) @ gaussians.instance_embedding.T).softmax(-1)
        instance_loss = F.cross_entropy(pred_instance_prob, instance_masks_batch)
        # semantic loss
        semantic_masks_batch = semantic_masks.reshape(h * w)[pixel_index]
        # semantic_render_feature = render_feature[:feature_dim, ...]
        # semantic_render_feature_batch = gaussians.semantic_decoder(instance_render_feature_batch)
        semantic_render_feature_batch = rendered_feature[gaussians.semantic_feature_dim:].reshape(-1, h * w).permute(1, 0)[pixel_index]
        # semantic_render_feature_batch = semantic_render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
        # semantic_loss = F.l1_loss(semantic_render_feature_batch, semantic_embedding_low_dim[semantic_masks_batch].detach())
        pred_semantic_prob = (semantic_render_feature_batch @ gaussians.semantic_embeddings.T).softmax(-1)
        # pred_semantic_prob = (gaussians.semantic_decoder(render_feature_batch) @ semantic_embedding_low_dim.T).softmax(-1)
        semantic_loss = F.cross_entropy(pred_semantic_prob, semantic_masks_batch)
       
       
        #norm loss
        # norm_loss = torch.norm(gaussians.instance_embeddings, dim=-1).mean() + torch.norm(semantic_embedding_low_dim, dim=-1).mean()
        
        loss = instance_contrastive_loss + semantic_contrastive_loss + semantic_loss + instance_loss
        
        # print(torch.sum(masks))
        # print(masks.shape)
        # print(torch.sum(masks[index].float()))
        # with autocast():
        # pixel_index = torch.randint(0, h * w, (batch_size, ))
        # pred_instance_label = gaussians.mask_decoder(render_feature.reshape(-1, h * w).permute(1, 0)[pixel_index, :])
        # pred_instance_label = gaussians.mask_decoder(render_feature.reshape(-1, h * w).permute(1, 0))
        # loss = ce_loss_fn(pred_instance_label, masks.reshape(h * w)[pixel_index])
        # loss = F.binary_cross_entropy_with_logits(pred_instance_label.reshape(-1), masks.reshape(h * w)[pixel_index].float())
        loss.backward()
        # scaler.scale(loss).backward()
        # total_loss += loss
        # loss.backward()

        iter_end.record()
        # scaler.scale(total_loss).backward()
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
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 20_000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument('--feature_gs', action='store_true', default=False)
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    # parser.add_argument("--images", type=str, default='images_4', required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    # print("Optimizing " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # # Start GUI server, configure and run training
    # # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations)

    # All done
    print("\nTraining complete.")
