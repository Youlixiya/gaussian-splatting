import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from utils.color import generate_contrasting_colors

COLORS = generate_contrasting_colors()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True)
    parser.add_argument("--refine_path", type=str, required=True)
    args = parser.parse_args()
    
    src_path = os.path.join(args.src_path, 'Annotations')
    refine_path = os.path.join(args.refine_path, 'Annotations')
    save_path = os.path.join(args.src_path, 'Annotations_fusion')
    os.makedirs(save_path, exist_ok=True)
    mask_names = os.listdir(src_path)
    src_masks_path = []
    refine_masks_path = []
    src_masks = []
    refine_masks = []
    
    for mask_name in mask_names:
        scr_mask_path = os.path.join(src_path, mask_name)
        src_masks_path.append(scr_mask_path)
        refine_mask_path = os.path.join(refine_path, mask_name)
        refine_masks_path.append(refine_mask_path)
        src_masks.append(cv2.cvtColor(cv2.imread(scr_mask_path), cv2.COLOR_BGR2RGB))
        refine_masks.append(cv2.cvtColor(cv2.imread(refine_mask_path), cv2.COLOR_BGR2RGB))
    
    src_masks = torch.from_numpy(np.stack(src_masks)).to(torch.uint8).cuda()
    refine_masks = torch.from_numpy(np.stack(refine_masks)).to(torch.uint8).cuda()
    
    src_colors = []
    refine_colors = []
    fusion_colors = []
    
    for i in tqdm(range(len(src_masks))):
        src_colors += torch.unique(src_masks[i].reshape(-1, 3), dim=0).tolist()
        refine_colors += torch.unique(refine_masks[i].reshape(-1, 3), dim=0).tolist()
        
    src_colors = torch.unique(torch.tensor(src_colors, dtype=torch.uint8, device='cuda'), dim=0)
    refine_colors = torch.unique(torch.tensor(refine_colors, dtype=torch.uint8, device='cuda'), dim=0)[[1], ...]
    for refine_color in refine_colors:
        i = 0
        while refine_color in src_colors:
        # while (refine_color in src_colors):
            refine_color = torch.tensor(COLORS[i], dtype=torch.uint8, device='cuda')
            i += 1
        fusion_colors.append(refine_color)
    
    for i in tqdm(range(len(src_masks))):
        src_mask = src_masks[i]
        refine_mask = refine_masks[i]
        for j, refine_color in enumerate(refine_colors):
            h, w = refine_mask.shape[:2]
            refine_mask = refine_mask.reshape(h*w, -1)
            target_color = refine_color[None, :].expand(refine_mask.size())
            mask = torch.eq(refine_mask, target_color).all(dim=1).reshape(h, w)
            # print(mask.shape)
            # print(src_mask.dtype)
            # print(fusion_colors[j].dtype)
            # print(src_mask[mask, :].shape)
            # mask = torch.equal(refine_mask.reshape(h*w, -1), refine_color).all(-1).reshape(h, w)
            src_mask[mask, :] = fusion_colors[j]
        # fusion_mask.append(src_mask)
        Image.fromarray(src_mask.cpu().numpy()).save(os.path.join(save_path, mask_names[i]))
    
            
    
    
    