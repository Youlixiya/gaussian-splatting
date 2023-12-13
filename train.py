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

    bg_color = [0] * gaussians.feature_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    loss_for_log = 0.0
    downscale = '' if args.images == 'images' else args.images.split('_')[-1]
    downscale = downscale if downscale == '' else f'_{downscale}'
    mask_dataset = MaskDataset(args.colmap_dir, scene.cameras.copy(), mask_dir=f'masks{downscale}')
    print(f'instance num: {len(mask_dataset.instance_colors)}')
    print(f'sematic num: {len(mask_dataset.sematic_colors)}')
    # gaussians.set_mask_decoder(len(mask_dataset.unique_colors))
    gaussians.set_instance_embeddings(len(mask_dataset.instance_colors))
    gaussians.set_sematic_embeddings(mask_dataset.clip_embeddings)
    gaussians.set_sematic_compressor(len(mask_dataset.sematic_colors))
    gaussians.set_instance_colors(mask_dataset.instance_colors, mask_dataset.sematic_colors)
    gaussians.feature_training_setup(opt)
    feature_dim = gaussians.feature_dim
    # fuse_sematic_table_bar = tqdm(range(len(mask_dataset)), desc="Fuse Sematic table")
    progress_bar = tqdm(range(cur_iter, opt.feature_iterations), desc="Training Feature GS progress")
    l1_loss_fn = nn.MSELoss()
    batch_size = 4096
    # scaler = GradScaler()
    # temperature = 100
    while cur_iter < opt.feature_iterations:
        cur_iter += 1
        iter_start.record()
        index = randint(0, len(mask_dataset)-1)
        viewpoint_stack = scene.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(index)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_feature=True)
        render_feature = render_pkg['render_feature']
        h, w = render_feature.shape[1:]
        instance_masks = mask_dataset.instance_masks[index]
        sematic_masks = mask_dataset.sematic_masks[index]
        pixel_index = torch.randint(0, h * w, (batch_size, ))
        # instance contrastive loss
        instance_contrastive_matrix = (gaussians.instance_embeddings @ gaussians.instance_embeddings.T).softmax(-1)
        gt_instance_contrastive_labels = torch.arange(gaussians.instance_embeddings.shape[0], device='cuda')
        instance_contrastive_loss = F.cross_entropy(instance_contrastive_matrix, gt_instance_contrastive_labels)
        # sematic contrastive loss
        sematic_embedding_low_dim = gaussians.sematic_compressor(gaussians.sematic_embeddings.float() * gaussians.sematic_scale)
        sematic_contrastive_matrix = (sematic_embedding_low_dim @ sematic_embedding_low_dim.T).softmax(-1)
        gt_sematic_contrastive_labels = torch.arange(sematic_embedding_low_dim.shape[0], device='cuda')
        sematic_contrastive_loss = F.cross_entropy(sematic_contrastive_matrix, gt_sematic_contrastive_labels)
        
        # render_feature_batch = render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
        
        # sematic loss
        sematic_masks_batch = sematic_masks.reshape(h * w)[pixel_index]
        # sematic_render_feature = render_feature[:feature_dim, ...]
        sematic_render_feature_batch = gaussians.sematic_decoder(render_feature.reshape(-1, h * w).permute(1, 0)[pixel_index])
        # sematic_render_feature_batch = sematic_render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
        # sematic_loss = F.l1_loss(sematic_render_feature_batch, sematic_embedding_low_dim[sematic_masks_batch].detach())
        pred_sematic_prob = (sematic_render_feature_batch @ sematic_embedding_low_dim.T).softmax(-1)
        # pred_sematic_prob = (gaussians.sematic_decoder(render_feature_batch) @ sematic_embedding_low_dim.T).softmax(-1)
        sematic_loss = F.cross_entropy(pred_sematic_prob, sematic_masks_batch)
        #instance loss
        instance_masks_batch = instance_masks.reshape(h * w)[pixel_index]
        instance_render_feature_batch = gaussians.instance_decoder(render_feature.reshape(-1, h * w).permute(1, 0)[pixel_index])
        # instance_render_feature = render_feature[feature_dim:, ...]
        # instance_render_feature_batch = instance_render_feature.reshape(-1, h * w)[:, pixel_index].permute(1, 0)
        # instance_loss = F.l1_loss(instance_render_feature_batch, gaussians.instance_embeddings[instance_masks_batch].detach())
        pred_instance_prob = (instance_render_feature_batch @ gaussians.instance_embeddings.T).softmax(-1)
        # pred_instance_prob = (gaussians.instance_decoder(render_feature_batch) @ gaussians.instance_embedding.T).softmax(-1)
        instance_loss = F.cross_entropy(pred_instance_prob, instance_masks_batch)
        #norm loss
        # norm_loss = torch.norm(gaussians.instance_embeddings, dim=-1).mean() + torch.norm(sematic_embedding_low_dim, dim=-1).mean()
        
        loss = instance_contrastive_loss + sematic_contrastive_loss + sematic_loss + instance_loss
        
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

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if cur_iter + 1 < opt.feature_iterations:
                gaussians.optimizer.step()
                # scaler.step(gaussians.optimizer)
                gaussians.optimizer.zero_grad(set_to_none = True)
                # scaler.update()

                # if (cur_iter + 1 in checkpoint_iterations):
                #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
                #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # total_loss /=  len(masks_embeddings) 
        # total_loss.backward()
            
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
