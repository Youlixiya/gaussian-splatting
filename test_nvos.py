import os
import cv2
import torch
import json
import numpy as np
import torch.nn.functional as F
from attrdict import AttrDict
from PIL import Image
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from utils.colormaps import ColormapOptions, apply_colormap
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

COLORS = generate_contrasting_colors()

def point_instance_segmentation(image, gaussian, points, render_instance_feature, mask_threshold, device='cuda'):
    h, w = render_instance_feature.shape[1:]
    points = torch.tensor(points, dtype=torch.int64, device=device)
    instance_embeddings = []
    for point in points:
        instance_embedding = F.normalize(render_instance_feature[:, point[0], point[1]][None], dim=-1)
        instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
        instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
        instance_embeddings.append(instance_embedding)
    instance_embeddings = torch.stack(instance_embeddings)
    similarity_map = (F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = image.clone()
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + torch.tensor(COLORS[i], dtype=torch.float32, device=device) /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map

def instance_segmentation_all(gaussian, render_instance_feature):
    h, w = render_instance_feature.shape[1:]
    instance_index = torch.argmax((F.normalize(render_instance_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
    # print(instance_index)
    instance_masks = gaussian.instance_colors[instance_index].reshape(h, w, 3)
    return instance_masks

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = AttrDict(json.load(f)[args.scene])
    args = AttrDict(args.__dict__)
    args.update(cfg)
    if 'rgb' in args.feature_gs_source:
        rgb_decode = True
    else:
        rgb_decode = False
    if 'depth' in args.feature_gs_source:
        depth_decode = True
    else:
        depth_decode = False
    gaussian = GaussianFeatureModel(sh_degree=3, rgb_decode=rgb_decode, depth_decode=depth_decode)
    gaussian.load_ply(args.gs_source)
    if args.feature_gs_source:
        gaussian.load_feature_params(args.feature_gs_source)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    feature_bg = torch.tensor([0] *gaussian.instance_feature_dim, dtype=torch.float32, device="cuda")
    colmap_cameras = None
    render_cameras = None
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        scene = CamScene(args.colmap_dir, h=h, w=w)
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras
        img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    for i, img in enumerate(imgs_name):
        if args.image_name == img:
            break
    cam = colmap_cameras.pop(i)
    os.makedirs(args.save_path, exist_ok=True)
    with torch.no_grad():
        render_pkg = render(cam, gaussian, pipe, background)
        image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
        image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
        render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.instance_features)['render_feature']
        total_rendered_feature = [render_feature]
        if gaussian.rgb_decode:
            total_rendered_feature.append(render_pkg['render'])
        if gaussian.depth_decode:
            total_rendered_feature.append(render_pkg['depth_3dgs'])
        total_rendered_feature = torch.cat(total_rendered_feature, dim=0)
        h, w = total_rendered_feature.shape[1:]
        total_rendered_feature = total_rendered_feature.reshape(-1, h*w).permute(1, 0)
        if gaussian.feature_aggregator:
            total_rendered_feature = F.normalize(gaussian.feature_aggregator(total_rendered_feature), dim=-1)
        else:
            total_rendered_feature = F.normalize(total_rendered_feature, dim=-1)
        total_rendered_feature = total_rendered_feature.permute(1, 0).reshape(-1, h, w)
        masks_all_instance, instance_mask_map, instance_object_map = point_instance_segmentation(image_tensor, gaussian, args.points, total_rendered_feature, args.mask_threshold, device='cuda')
        instance_masks = instance_segmentation_all(gaussian, total_rendered_feature)
        image.save(os.path.join(args.save_path, f'rendered_rgb_{args.image_name}'))
        Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'rendered_feature_pca_{args.image_name}'))
        Image.fromarray((apply_colormap(total_rendered_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'total_rendered_feature_pca_{args.image_name}'))
        Image.fromarray(np.stack([(masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(args.save_path, args.mask_save_name))
        Image.fromarray((instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_mask_map_{args.image_name}'))
        Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_object_map_{args.image_name}'))
        Image.fromarray((instance_masks.cpu().numpy()).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_masks_{args.image_name}'))
    # device = "cuda:0"
    # self.colors = np.random.random((500, 3))