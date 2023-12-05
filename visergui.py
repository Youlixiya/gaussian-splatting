from threading import Thread
import torch
import numpy as np
from PIL import Image
import os
import time
import tqdm
import argparse
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from pprint import pprint
# from utils import qvec2rotmat
import cv2
# from utils import Timer
import torchvision.transforms as T
from collections import deque, defaultdict
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from render import Renderer
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

class Timer:
    recorder = defaultdict(list)

    def __init__(self, des="", verbose=False, record=True, debug=True) -> None:
        self.des = des
        self.verbose = verbose
        self.record = record
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return self
        self.start = time.time()
        self.start_cuda = torch.cuda.Event(enable_timing=True)
        self.end_cuda = torch.cuda.Event(enable_timing=True)
        self.start_cuda.record()
        return self

    def __exit__(self, *args):
        if not self.debug:
            return 
        self.end = time.time()
        self.end_cuda.record()
        torch.cuda.synchronize()
        self.interval = self.start_cuda.elapsed_time(self.end_cuda)/1000.
        if self.verbose:
            print(f"[cudasync]{self.des} consuming {self.interval:.8f}")
        if self.record:
            Timer.recorder[self.des].append(self.interval)

    @staticmethod
    def show_recorder():
        pprint({k: np.mean(v) for k, v in Timer.recorder.items()})

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

# def c2wtow2c(c2w):
#     R = c2w[:3, :3].T
#     t = -R @ c2w[:3, 3]
#     w2c = np.zeros((3, 4))
#     w2c[:3, :3] = R
#     w2c[:3, 3] = t
#     return w2c

def c2wtow2c(c2w):
    return np.linalg.inv(c2w)

def nearest_cameras(poses, target_pose, num_cam=2):
    distances = torch.norm(poses.reshape(poses.shape[0], -1) - target_pose.reshape(1, -1), dim=-1)
    poses_index = torch.argsort(distances)[:num_cam]
    return poses[poses_index, ...], poses_index

class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, opt, views, device, viewer_port):
        self.opt = opt
        self.views = views
        self.device = device
        self.port = viewer_port
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button("Reset View")

        self.need_update = False
        # self.llava_model_path = "liuhaotian/llava-v1.5-7b"
        # self.llava_model_name=get_model_name_from_path(self.llava_model_path)
        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #     model_path=self.llava_model_path,
        #     model_base=None,
        #     model_name=self.llava_model_name,
        #     load_4bit=True,
        #     device_map='',
        #     device='cuda:0',
        # )
        self.outputs = None
        self.object_name = []
        self.mask_trigs = []
        # self.xyz = None
        self.mask_2d = None
        self.mask_3d = []
        self.point_3d = []
        # self.llava_chat = False
        # self.get_2dmask = False
        
        
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )
        self.views_slider = self.server.add_gui_slider(
            "View Index", min=0, max=len(views), step=len(views), initial_value=0
        )
        self.show_mode = self.server.add_gui_dropdown('Show Mode', options=['box', 'mask2d', 'mask3d'], initial_value='mask2d')
        # self.llava_chat = self.server.add_gui_button("LLAVA Chat")
        
        self.text_prompt = self.server.add_gui_text("Text Prompt", initial_value='')
        self.mask2d_bu = self.server.add_gui_button("2DMASK")
        self.proj_mask_to_3d_bu = self.server.add_gui_button("Proj MASK To 3D")
        self.clear_mask_3d_bu = self.server.add_gui_button("Clear MASK3D")
        # self.text_output = self.server.add_gui_text("Text Output", initial_value='')
        
        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )
        
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True
        
        @self.views_slider.on_update
        def _(_):
            self.need_update = True
            
        @self.show_mode.on_update
        def _(_):
            self.need_update = True
            
        @self.mask2d_bu.on_click
        def _(_):
            self.need_update = True
        
        @self.proj_mask_to_3d_bu.on_click
        def _(_):
            self.proj_mask_to_3d()
        
        @self.clear_mask_3d_bu.on_click
        def _(_):
            self.clear_3D_mask()
        
        @self.text_prompt.on_update
        def _(_):
            self.need_update = True

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer
    
    @torch.no_grad()
    def update(self):
        # if self.proj_mask_to_3d_bu.value:
        #     self.proj_mask_to_3d()
        # if self.clear_mask_3d_bu.value:
        #     self.clear_3D_mask()
        if self.need_update:
            start = time.time()
            for client in self.server.get_clients().values():
                camera = client.camera
                W = self.resolution_slider.value
                H = int(self.resolution_slider.value/camera.aspect)
                znear = self.near_plane_slider.value
                zfar = self.far_plane_slider.value
                world_view_transform = torch.tensor(get_w2c(camera)).cuda()
                world_view_transform[:3, [1, 2]] *= -1
                world_view_transform = world_view_transform.transpose(0, 1)
                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=camera.fov, fovY=camera.fov).transpose(0,1).cuda()
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                mincam = MiniCam(W, H, camera.fov, camera.fov, znear, zfar, world_view_transform, full_proj_transform)
                # cam = self.views[int(self.views_slider.value)]
                # c2w = torch.from_numpy(get_c2w(camera))
                # c2w[:3, [1, 2]] *= -1
                # self.pose = c2w
                # w2c[:3, [1, 2]] *= -1
                # c2w[:, 0] *= -1
                
                try:
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()
                    self.outputs = self.renderer.render(
                        mincam
                    )
                    # if self.llava_chat.value:
                    #     self.get_object_name_from_implicit_query(outputs)
                    #     self.llava_chat.value = False
                    #     out = outputs["image"].astype(np.float32)
                    out = self.outputs["render"].permute(1, 2, 0).cpu().numpy()
                    # print(out.shape)
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.
                    
                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue
                client.set_background_image(out, format="jpeg")
                # client.add_image(name='render_result', image=out, format="jpeg", render_width=W, render_height=H)
                
                self.debug_idx += 1
                # if self.debug_idx % 100 == 0:
                #     cv2.imwrite(
                #         f"./tmp/viewer/debug_{self.debug_idx}.png",
                #         cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                #     )

            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            # print(f"Update time: {end - start:.3g}")

if __name__ == '__main__':
    from scene import Scene, GaussianModel
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    gaussians = GaussianModel(model.extract(args).sh_degree)
    # gaussians.load_ply(os.path.join(args.model_path,
    #                                 "point_cloud",
    #                                 "iteration_" + str(args.iteration),
    #                                 "point_cloud.ply"))
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    views = scene.getTrainCameras()
    renderer = Renderer(gaussians, pipeline.extract(args))
    webui = ViserViewer(args, views, device='cuda', viewer_port=6789)
    webui.set_renderer(renderer)
    while(True):
        webui.update()
