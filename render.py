import os
import cv2
import torch
import json
import clip
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.general_utils import AttrDict
from PIL import Image
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from scene import Scene, GaussianModel, GaussianFeatureModel
from utils.colormaps import ColormapOptions, apply_colormap, get_pca_dict
from utils.color import generate_contrasting_colors
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from lisa.lisa_pipeline import LISAPipeline

COLORS = torch.tensor(generate_contrasting_colors(500), dtype=torch.uint8, device='cuda')

@torch.no_grad()
def select_semantic_embeddings(clip_model, gaussian, text_prompt, neg_features, text_query_threshold, device='cuda'):
    
    text_prompt = clip.tokenize(text_prompt.split(',')).to(device)
    # text_prompt = alpha_clip.tokenize([self.text_prompt.value]).to(self.device)
    pos_features = clip_model.encode_text(text_prompt)
    pos_features /= pos_features.norm(dim=-1, keepdim=True)
    
    total_features = torch.cat([neg_features, pos_features])
    total_mm = gaussian.clip_embeddings @ total_features.T
    pos_mm = total_mm[:, 1:]
    neg_mm = total_mm[:, [0]].repeat(1, pos_mm.shape[-1])
    # print(pos_mm.shape)
    # print(pos_mm.shape)
    total_similarity = torch.stack([pos_mm, neg_mm], dim=-1)
    softmax = (100 * total_similarity).softmax(-1)
    pos_softmax = softmax[..., 0]
    valid_mask = pos_softmax > text_query_threshold
    semantic_valid_num = valid_mask.sum(0)
    semantic_embeddings = []
    for i in range(valid_mask.shape[-1]):
        semantic_embeddings.append(gaussian.instance_embeddings[valid_mask[:, i], :])
    semantic_embeddings = torch.cat(semantic_embeddings)
    return semantic_valid_num, semantic_embeddings

def get_instance_embeddings(gaussian, points, instance_feature, device='cuda'):
    h, w = instance_feature.shape[1:]
    points = torch.tensor(points, dtype=torch.int64, device=device)
    instance_embeddings = []
    for point in points:
        instance_embedding = instance_feature[:, point[0], point[1]][None]
        instance_embedding_index = torch.argmax((instance_embedding @ gaussian.instance_embeddings.T).softmax(-1))
        instance_embedding = gaussian.instance_embeddings[instance_embedding_index]
        instance_embeddings.append(instance_embedding)
    instance_embeddings = torch.stack(instance_embeddings)
    return instance_embeddings

def get_lisa_instance_embeddings(lisa_pipeline, gaussian, lisa_text_prompt, image, instance_feature, device='cuda'):
    result_list, mask_result_list, mask_list, mask_rgb_list, output_str = lisa_pipeline(lisa_text_prompt, image=image)
    lisa_mask = mask_list[0]

    lisa_mask = torch.tensor(lisa_mask, dtype=torch.bool, device=device)
    mask_instance_feature = instance_feature[:, lisa_mask].permute(1, 0)
    instance_embeddings_index = torch.argmax(mask_instance_feature @ gaussian.instance_embeddings.T, dim=-1)
    unique_index, counts = torch.unique(instance_embeddings_index, return_counts=True)
    instance_embedding = gaussian.instance_embeddings[unique_index[torch.argmax(counts)]]
    return result_list[0], output_str, instance_embedding

def point_instance_segmentation(image, instance_embeddings, instance_feature, mask_threshold, device='cuda'):
    similarity_map = (instance_feature.reshape(-1, h * w).permute(1, 0) @ instance_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all_instance = masks.any(-1)
    instance_mask_map = image.clone()
    instance_object_map = torch.cat([image.clone(), masks_all_instance[...,None]], dim=-1)
    # instance_object_map = torch.stack([instance_object_map, torch.full_like(instance_object_map, 0, device='cuda')], dim=-1)
    for i, mask in enumerate(masks.permute(2, 0, 1)):
        instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + COLORS[i] /255 * 0.5
    instance_mask_map[~masks_all_instance, :] /= 2
    # instance_object_map[masks_all_instance, -1] = torch.tensor([1], dtype=torch.float32, device=device)
    
    return masks_all_instance, instance_mask_map, instance_object_map

def text_semantic_segmentation(image, semantic_embeddings, instance_feature, mask_threshold, semantic_valid_num=None, device='cuda'):
    similarity_map = (instance_feature.reshape(-1, h * w).permute(1, 0) @ semantic_embeddings.T).reshape(h, w, -1)
    masks = (similarity_map > mask_threshold)
    masks_all = masks.any(-1)
    semantic_mask_map = image.clone()
    # sematic_object_map = image.clone()
    sematic_object_map = torch.cat([image.clone(), masks_all[..., None]], dim=-1)
    start_index = 0
    # print(semantic_valid_num)
    # print(semantic_embeddings.shape)
    for i in range(len(semantic_valid_num)):
        mask = masks[..., start_index:start_index + semantic_valid_num[i]].any(-1)
        # print(mask.shape)
        semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + COLORS[i] / 255 * 0.5
        # semantic_mask_map[~mask, :] = semantic_mask_map[~mask, :] * 0.5 + torch.tensor([0, 0, 0], device=self.device) * 0.5
        start_index += semantic_valid_num[i]
    semantic_mask_map[~masks_all, :] /= 2
    # sematic_object_map[~masks_all, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    # sematic_object_map[masks_all, -1] = torch.tensor([1], dtype=torch.float32, device=device)
    
    return masks_all, semantic_mask_map, sematic_object_map

def instance_segmentation_all(image, gaussian, instance_feature):
    h, w = instance_feature.shape[1:]
    instance_index = torch.argmax((instance_feature.reshape(-1, h * w).permute(1, 0) @ gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
    # print(instance_index)
    instance_masks = COLORS[instance_index].reshape(h, w, 3) /255 * 0.5 + image * 0.5
    return instance_masks

if __name__ == '__main__':
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pipe = PipelineParams(parser)
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--clip", action='store_true')
    parser.add_argument("--lisa", action='store_true')
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = AttrDict(json.load(f)[args.scene])
    args = AttrDict(args.__dict__)
    args.update(cfg)
    # if 'rgb' in args.feature_gs_source:
    #     rgb_decode = True
    # else:
    #     rgb_decode = False
    # if 'depth' in args.feature_gs_source:
    #     depth_decode = True
    # else:
    #     depth_decode = False
    gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=args.gs_feature_dim)
    gaussian.load_ply(args.gs_source)
    if args.feature_gs_source:
        gaussian.load_feature_params(args.feature_gs_source)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    feature_bg = torch.tensor([0] *gaussian.gs_feature_dim, dtype=torch.float32, device="cuda")
    colmap_cameras = None
    render_cameras = None
    #clip
    clip_model, preprocess = clip.load(args.clip_model_type, mask_prompt_depth=3, device='cuda')
    clip_ckpt = torch.load(cfg.clip_model_pretrained)
    clip_model.load_state_dict(clip_ckpt)
    neg_features = clip_model.encode_text(clip.tokenize(['background']).to('cuda'))
    neg_features /= neg_features.norm(dim=-1, keepdim=True)
    neg_features = neg_features
    #lisa
    if args.lisa:
        lisa_pipeline = LISAPipeline(args.lisa_model_type, local_rank=0, load_in_4bit=False, load_in_8bit=True, conv_type=args.lisa_conv_type)
        lisa_instance_embeddings = None
        lisa_mask_map_save_path = os.path.join(args.save_path, 'lisa', 'mask_maps')
        lisa_mask_save_path = os.path.join(args.save_path, 'lisa', 'masks')
        lisa_mask_object_save_path = os.path.join(args.save_path, 'lisa', 'object')
        os.makedirs(lisa_mask_map_save_path, exist_ok=True)
        os.makedirs(lisa_mask_save_path, exist_ok=True)
        os.makedirs(lisa_mask_object_save_path, exist_ok=True)
    if args.colmap_dir is not None:
        img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
        h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
        scene = CamScene(args.colmap_dir, h=h, w=w)
        cameras_extent = scene.cameras_extent
        colmap_cameras = scene.cameras
        img_suffix = os.listdir(os.path.join(args.colmap_dir, args.images))[0].split('.')[-1]
        imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in colmap_cameras]
        imgs_path = [os.path.join(args.colmap_dir, args.images, img_name) for img_name in imgs_name]
    point_instance_embeddings = None
    clip_semantic_embeddings = None
    
    rendered_feature_pca_dict = None
    instance_feature_pca_dict = None
    point_mask_map_save_path = os.path.join(args.save_path, 'point', 'mask_maps')
    point_mask_save_path = os.path.join(args.save_path, 'point', 'masks')
    point_mask_object_save_path = os.path.join(args.save_path, 'point', 'object')
    clip_mask_map_save_path = os.path.join(args.save_path, 'clip', 'mask_maps')
    clip_mask_save_path = os.path.join(args.save_path, 'clip', 'masks')
    clip_mask_object_save_path = os.path.join(args.save_path, 'clip', 'object')
    
    anything_mask_save_path = os.path.join(args.save_path, 'anything', 'mask_maps')
    rgb_save_path = os.path.join(args.save_path, 'rgb')
    rendered_feature_save_path = os.path.join(args.save_path, 'rendered_feature')
    instance_feature_save_path = os.path.join(args.save_path, 'instance_feature')
    os.makedirs(point_mask_map_save_path, exist_ok=True)
    os.makedirs(point_mask_save_path, exist_ok=True)
    os.makedirs(point_mask_object_save_path, exist_ok=True)
    os.makedirs(clip_mask_map_save_path, exist_ok=True)
    os.makedirs(clip_mask_save_path, exist_ok=True)
    os.makedirs(clip_mask_object_save_path, exist_ok=True)
    os.makedirs(anything_mask_save_path, exist_ok=True)
    os.makedirs(rgb_save_path, exist_ok=True)
    os.makedirs(rendered_feature_save_path, exist_ok=True)
    os.makedirs(instance_feature_save_path, exist_ok=True)
    for i in tqdm(range(len(colmap_cameras))):
        cam = colmap_cameras[i]
        with torch.no_grad():
            render_pkg = render(cam, gaussian, pipe, background)
            image_tensor = render_pkg['render'].permute(1, 2, 0).clamp(0, 1)
            image = Image.fromarray((image_tensor.cpu().numpy() * 255).astype(np.uint8))
            render_feature = render(cam, gaussian, pipe, feature_bg, render_feature=True, override_feature=gaussian.gs_features)['render_feature']
            instance_feature = F.normalize(render_feature, dim=0)
            if rendered_feature_pca_dict is None:
                rendered_feature_pca_dict = get_pca_dict(render_feature)
            if instance_feature_pca_dict is None:
                instance_feature_pca_dict = get_pca_dict(instance_feature)
            if clip_semantic_embeddings == None:
                semantic_valid_num, clip_semantic_embeddings = select_semantic_embeddings(clip_model, gaussian, args.clip_text_prompt, neg_features, args.text_query_threshold) 
            if point_instance_embeddings == None:
                point_instance_embeddings = get_instance_embeddings(gaussian, args.points, instance_feature)
            if args.lisa:
                if lisa_instance_embeddings == None:
                    lisa_mask, lisa_output, lisa_instance_embeddings = get_lisa_instance_embeddings(lisa_pipeline, gaussian, args.lisa_text_prompt, image, instance_feature)
                    Image.fromarray(lisa_mask).save(os.path.join(args.save_path, 'lisa_mask.jpg'))
                    with open(os.path.join(args.save_path, 'lisa_text_output.txt'), 'w') as f:
                        f.write(f'Input: {args.lisa_text_prompt}\nOutput: {lisa_output}')
                lisa_masks_all_instance, lisa_instance_mask_map, lisa_instance_object_map = point_instance_segmentation(image_tensor, lisa_instance_embeddings, instance_feature, args.mask_threshold, device='cuda')
                Image.fromarray(np.stack([(lisa_masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(lisa_mask_save_path, f'mask_{i:03d}.png'))
                Image.fromarray((lisa_instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(lisa_mask_map_save_path, f'mask_map_{i:03d}.jpg'))
                Image.fromarray((lisa_instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGBA").save(os.path.join(lisa_mask_object_save_path, f'mask_object_{i:03d}.png'))
            point_masks_all_instance, point_instance_mask_map, point_instance_object_map = point_instance_segmentation(image_tensor, point_instance_embeddings, instance_feature, args.mask_threshold, device='cuda')
            masks_all_semantic, semantic_mask_map, semantic_object_map = text_semantic_segmentation(image_tensor, clip_semantic_embeddings, instance_feature, args.mask_threshold, semantic_valid_num)
            
            anything_mask = instance_segmentation_all(image_tensor, gaussian, instance_feature)
            # image.save(os.path.join(args.save_path, f'rendered_rgb_{args.image_name}'))
            # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'rendered_feature_pca_{args.image_name}'))
            # Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca")).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'total_rendered_feature_pca_{args.image_name}'))
            image.save(os.path.join(rgb_save_path, f'rgb_{i:03d}.jpg'))
            Image.fromarray((apply_colormap(render_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=rendered_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(rendered_feature_save_path, f'rendered_feature_pca_{i:03d}.jpg'))
            Image.fromarray((apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca", pca_dict=instance_feature_pca_dict)).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(instance_feature_save_path, f'instance_feature_pca_{i:03d}.jpg'))
            Image.fromarray(np.stack([(point_masks_all_instance.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(point_mask_save_path, f'mask_{i:03d}.png'))
            Image.fromarray((point_instance_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(point_mask_map_save_path, f'mask_map_{i:03d}.jpg'))
            Image.fromarray((point_instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGBA").save(os.path.join(point_mask_object_save_path, f'mask_object_{i:03d}.png'))
            Image.fromarray(np.stack([(masks_all_semantic.cpu().numpy() * 255).astype(np.uint8)] * 3, axis=-1)).save(os.path.join(clip_mask_save_path, f'mask_{i:03d}.png'))
            Image.fromarray((semantic_mask_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(clip_mask_map_save_path, f'mask_map_{i:03d}.jpg'))
            Image.fromarray((semantic_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGBA").save(os.path.join(clip_mask_object_save_path, f'mask_object_{i:03d}.png'))
            
            Image.fromarray((anything_mask.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(anything_mask_save_path, f'mask_map_{i:03d}.jpg'))
            # Image.fromarray((instance_object_map.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(args.save_path, f'instance_object_map_{args.image_name}'))
            # Image.fromarray((instance_masks.cpu().numpy()).astype(np.uint8)).save(os.path.join(mask_save_path, f'mask_{i}.png'))
    # device = "cuda:0"
    # self.colors = np.random.random((500, 3))