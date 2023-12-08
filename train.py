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
import torch
from torch import nn
import torch.nn.functional as F
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

    gaussians.feature_training_setup(opt)

    bg_color = [0] * gaussians.feature_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    loss_for_log = 0.0
    progress_bar = tqdm(range(cur_iter, opt.feature_iterations), desc="Training Feature GS progress")
    mask_dataset = MaskDataset(args.colmap_dir, scene.cameras.copy())
    bce_loss = nn.BCELoss()
    while cur_iter < opt.feature_iterations:
        index = randint(0, len(mask_dataset)-1)
        viewpoint_stack = scene.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(index)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, render_feature=True)
        render_feature = render_pkg['render_feature']
        h, w = render_feature.shape[1:]
        masks_embeddings = mask_dataset[index]
        total_loss = torch.empty(0).cuda()
        for i in range(len(masks_embeddings)):
            cur_iter += 1
            iter_start.record()
            gaussians.update_learning_rate(cur_iter)
            mask_embedding = masks_embeddings[i]
            mask = torch.tensor(mask_embedding['mask'], dtype=torch.bool, device='cuda').unsqueeze(0)
            embedding = mask_embedding['clip_embedding'].float().cuda()
            if gaussians.sematic_table.is_empty():
                gaussians.sematic_table.append(embedding)
            else:
                query_cosine_similarities = gaussians.sematic_table.get_cosine_similarities(embedding)
                max_index = torch.argmax(query_cosine_similarities)
                max_cosine_similaritie = query_cosine_similarities[max_index]
                if max_cosine_similaritie > 0.9:
                    # print(max_cosine_similaritie)
                    # print(max_index)
                    # print(gaussians.sematic_table.table[max_index].shape)
                    new_embedding = F.normalize((max_cosine_similaritie * gaussians.sematic_table.table[max_index] + (1 - max_cosine_similaritie) * embedding).unsqueeze(0)).squeeze(0)
                    # new_embedding = max_cosine_similaritie * gaussians.sematic_table.table[max_index] + (1 - max_cosine_similaritie) * embedding
                    # print(new_embedding.shape)
                    gaussians.sematic_table.table[max_index] = new_embedding
                    embedding = new_embedding
                else:
                    gaussians.sematic_table.append(embedding)
            embedding_low_dim = gaussians.sematic_compressor(embedding.unsqueeze(0)).squeeze(0)
            # print(render_feature.shape)
            # print(embedding_low_dim[None, :, None, None].shape)
            # embedding_low_dim_map = torch.stack(h * w * [embedding_low_dim], dim=-1).reshape(1, -1, h, w)
            # mask_decoder_input = render_feature.unsqueeze(0) + embedding_low_dim[None, :, None, None]
            # mask_decoder_input = torch.cat([render_feature.unsqueeze(0), embedding_low_dim_map], dim=1)
            # pred_mask = torch.sigmoid(gaussians.mask_decoder(mask_decoder_input).squeeze(0))
            # pred_mask = torch.sigmoid(gaussians.mask_decoder(render_feature.unsqueeze(0) + embedding_low_dim[None, :, None, None]).squeeze(0))
            pred_sematic = render_feature + gaussians.sematic_decoder(render_feature.unsqueeze(0) + embedding_low_dim[None, :, None, None]).squeeze(0)
            cosine_similarities_map = get_cosine_similarities(pred_sematic.reshape(-1, h * w).permute(1, 0), embedding_low_dim).reshape(h, w, 1).permute(2, 0, 1)
            pos_cosine_similarities_map = cosine_similarities_map[mask]
            neg_cosine_similarities_map = cosine_similarities_map[~mask]
            # pred_pos_mask = pred_mask[mask]
            # pred_neg_mask = pred_mask[~mask]
            # pred_pose_sematic = pred_sematic[mask.squeeze(0)]
            pos_mask = torch.ones_like(pos_cosine_similarities_map, device='cuda')
            # pos_mask = torch.ones_like(pred_pos_mask, device='cuda')
            neg_mask = torch.zeros_like(neg_cosine_similarities_map, device='cuda')
            
            cosine_similarities_loss = (pos_mask.reshape(-1) - pos_cosine_similarities_map.reshape(-1)).mean() + \
                                        ((neg_cosine_similarities_map.reshape(-1) - neg_mask.reshape(-1))).mean()
            # bce_loss = cosine_similarities_map.reshape(-1), mask.float().reshape(-1)
            # cosine_similarities_loss = (bce_loss(pos_cosine_similarities_map.reshape(-1), pos_mask.reshape(-1)).mean() + \
            #                             bce_loss(neg_cosine_similarities_map.reshape(-1), neg_mask.reshape(-1)).mean()) / 2
            
            # print(pos_mask.reshape(-1).shape)
            # print(pred_pos_mask.reshape(-1).shape)
            # print(neg_mask.reshape(-1).shape)
            # print(pred_neg_mask.reshape(-1).shape)
            # print(neg_cosine_similarities_map.reshape(-1).shape)
            
            
            # adaptive_bce_loss= (bce_loss(pos_mask.reshape(-1), pred_pos_mask.reshape(-1))).mean() 
                            #    (neg_cosine_similarities_map.reshape(-1) * bce_loss(neg_mask.reshape(-1), pred_neg_mask.reshape(-1))).mean()
            # feature_dim = gaussians.feature_dim
            # norm_loss = 1 - torch.norm(embedding_low_dim) + (pos_mask.reshape(-1) - torch.norm(pred_pose_sematic.reshape(feature_dim, -1), dim=-1).reshape(-1)).mean()
            # loss = cosine_similarities_loss + adaptive_bce_loss + norm_loss
            # loss = bce_loss
            loss = cosine_similarities_loss
            loss.backward(retain_graph=True)
            # total_loss += loss
            # loss.backward()

            iter_end.record()
            
            with torch.no_grad():
                # Progress bar
                loss_for_log = loss.item()
                # if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss_for_log:.{7}f}", "Sematic_table_len": len(gaussians.sematic_table)})
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
                    gaussians.optimizer.zero_grad(set_to_none = True)

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
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 10_000])
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
