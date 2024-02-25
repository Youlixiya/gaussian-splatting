from typing import Tuple
import numpy as np
import torch

def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    expand_ratio: float = 1.0,
    max_h: int = None,
    max_w: int = None,
):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = x2 - x1
    h = y2 - y1
    w = w * expand_ratio
    h = h * expand_ratio
    box = [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]
    if max_h is not None:
        box[1] = max(0, box[1])
        box[3] = min(max_h - 1, box[3])
    if max_w is not None:
        box[0] = max(0, box[0])
        box[2] = min(max_w - 1, box[2])
    return [int(b) for b in box]


# def mask2box(mask: torch.Tensor):
#     # use naive way
#     row = torch.nonzero(mask.sum(dim=0))[:, 0]
#     if len(row) == 0:
#         return None
#     x1 = row.min()
#     x2 = row.max()
#     col = np.nonzero(mask.sum(dim=1))[:, 0]
#     y1 = col.min()
#     y2 = col.max()
#     return x1, y1, x2 + 1, y2 + 1

def mask2box(mask: torch.Tensor):
    nonzero_indices = torch.nonzero(mask.float())
    x1, y1, x2, y2 = nonzero_indices[:, 1].min(), nonzero_indices[:, 0].min(), \
             nonzero_indices[:, 1].max() + 1, \
             nonzero_indices[:, 0].max() + 1
    # print(nonzero_indices[:, 1].min(), nonzero_indices[:, 0].min(), nonzero_indices[:, 1].max(), nonzero_indices[:, 0].max())
    return x1, y1, x2, y2


def crop_with_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    bbox: torch.Tensor,
    fill: Tuple[float, float, float] = (0, 0, 0),
    expand_ratio: float = 1.0,
):
    # l, t, r, b = expand_box(*bbox, expand_ratio)
    l, t, r, b = bbox
    _, h, w = image.shape
    l = max(l, 0)
    t = max(t, 0)
    r = min(r, w)
    b = min(b, h)
    # if b < t:
    #     b = min(t + 1, h)
    # if r < l:
    #     r = min(l + 1, h)
    new_image = torch.cat(
        [image.new_full((1, b - t, r - l), fill_value=val) for val in fill]
    )
    # return image[:, t:b, l:r], mask[None, t:b, l:r]
    return image[:, t:b, l:r] * mask[None, t:b, l:r] + (1 - mask[None, t:b, l:r]) * new_image, mask[None, t:b, l:r]


# def build_clip_model(model: str, mask_prompt_depth: int = 0, frozen: bool = True):
#         # download on rank 0 only
#     model, _ = clip.load(model, mask_prompt_depth=mask_prompt_depth, device="cpu")
#     if frozen:
#         for param in model.parameters():
#             param.requires_grad = False
#     return model