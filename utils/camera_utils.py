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
import torch
from scene.cameras import Camera, MiniCam, Simple_Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch.nn.functional as F

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def cameraList_load(cam_infos, h, w):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            Simple_Camera(colmap_id=c.uid, R=c.R, T=c.T,
                   FoVx=c.FovX, FoVy=c.FovY, h=h, w=w, qvec = c.qvec,
                   image_name=c.image_name, uid=id, data_device='cuda')
        )
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def project(camera: MiniCam, points3d):
    # TODO: should be equivalent to full_proj_transform.T
    if isinstance(points3d, list):
        points3d = torch.stack(points3d, dim=0)
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    xy = points3d_camera[..., :2] / points3d_camera[..., 2:]
    ij = (
        xy
        * torch.tensor(
            [
                fov2focal(camera.FoVx, camera.image_width),
                fov2focal(camera.FoVy, camera.image_height),
            ],
            dtype=torch.float32,
            device=xy.device,
        )
        + torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=xy.device,
        )
        / 2
    ).to(torch.long)

    return ij


def unproject(camera: MiniCam, points2d, depth):
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    points2d[0] *= camera.image_width
    points2d[1] *= camera.image_height
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)

    points3d = (
        directions * depth[0][points2d[..., 1], points2d[..., 0]] + origin[None, ...]
    )

    return points3d


def get_point_depth(points3d, camera: MiniCam):
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    depth = points3d_camera[..., 2:]
    return depth