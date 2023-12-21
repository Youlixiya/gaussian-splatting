import os
import json
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from attrdict import AttrDict
from PIL import Image
from argparse import ArgumentParser

def get_iou(pred_mask, gt_mask):
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    iou = intersection.sum() / union.sum()
    return iou

def get_accuracy(pred_mask, gt_mask):
    h, w = pred_mask.shape
    # print(((pred_mask==255) & (gt_mask==255)).shape)
    # print((pred_mask==255).sum())
    pos_acc = ((pred_mask==255) & (gt_mask==255)).sum() / (gt_mask==255).sum()
    # print(pos_acc)
    neg_acc = ((pred_mask==0) & (gt_mask==0)).sum() / (gt_mask==0).sum()
    # tp_fn = (pred_mask == gt_mask).sum()
    return (pos_acc + neg_acc) / 2

def get_pr(pred_mask, gt_mask):
    # tp = (pred_mask == gt_mask).sum()
    tp = ((pred_mask & gt_mask) == 255).sum()
    tp_fp = (pred_mask.reshape(-1) == 255).sum()
    return tp / tp_fp

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--save_tag", type=str, required=True)
    
    args = parser.parse_args()
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
    masks = []
    gt_masks = []
    scenes = []
    ious = []
    accs = []
    prs = []
    metrics = 'Scene\tIoU\tAcc\tPR\n'
    for scene, scene_cfg in cfg.items():
        scenes.append(scene)
        scene_cfg = AttrDict(scene_cfg)
        mask_path = os.path.join(scene_cfg.save_path, scene_cfg.mask_save_name)
        gt_mask_path = os.path.join(args.gt_path, scene, scene_cfg.mask_save_name)
        masks.append(Image.open(mask_path))
        gt_masks.append(Image.open(gt_mask_path))
    for i in tqdm(range(len(scenes))):
        mask = np.array(masks[i])[..., 0]
        h, w = mask.shape
        gt_mask = np.array(gt_masks[i].resize((w, h)))
        iou = get_iou(mask, gt_mask)
        acc = get_accuracy(mask, gt_mask)
        pr = get_pr(mask, gt_mask)
        ious.append(iou)
        accs.append(acc)
        prs.append(pr)
        metrics += f'{scenes[i]}\t{iou}\t{acc}\t{pr}\n'
    mean_iou = sum(ious) / len(ious)
    mean_acc = sum(accs) / len(accs)
    mean_pr = sum(prs) / len(prs)
    metrics += f'mean\t{mean_iou}\t{mean_acc}\t{mean_pr}'
    with open(f'{args.save_tag}_metrics.txt', 'w') as f:
        f.write(metrics)
        
        
    # args = AttrDict(args.__dict__)
    # args.update(cfg)