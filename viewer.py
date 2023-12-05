from typing import Any
import os
import torch
import torchvision
import random
import math
import numpy as np
from PIL import Image
import time
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from collections import deque
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.camera_utils import project, unproject
from utils.sam import LangSAMTextSegmentor
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel

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


# def get_c2w(camera):
#     c2w = np.zeros([4, 4], dtype=np.float32)
#     c2w[:3, :3] = qvec2rotmat(camera.wxyz).T
#     c2w[:3, 3] = camera.position
#     c2w[3, 3] = 1.0

#     c2w = torch.from_numpy(c2w).to("cuda")

    # return c2w

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class ViserViewer:
    def __init__(self, cfg):
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.gaussian = GaussianModel(sh_degree=3)
        self.gaussian.load_ply(self.gs_source)
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)
        # self.set_system(System(self.gaussian, pipe, background_tensor))
        self.device = "cuda:0"
        self.port = 8080

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        
        self.points3d = []
         # front end related
        self.colmap_cameras = None
        self.render_cameras = None
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}
        # self.text_segmentor = LangSAMTextSegmentor().cuda()
        # self.sam_predictor = self.text_segmentor.model.sam
        # self.sam_predictor.is_image_set = True
        self.sam_features = {}
        self.semantic_gauassian_masks = {}
        self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)
        
        with self.server.add_gui_folder("Semantic Tracing"):
            self.sam_enabled = self.server.add_gui_checkbox(
                "Enable SAM",
                initial_value=False,
            )
            self.add_sam_points = self.server.add_gui_checkbox(
                "Add SAM Points", initial_value=False
            )
        with self.server.add_gui_folder("Render Setting"):
            self.reset_view_button = self.server.add_gui_button("Reset View")

            self.toggle_axis = self.server.add_gui_checkbox(
                "Toggle Axis",
                initial_value=True,
            )

            self.need_update = False

            self.pause_training = False

            self.train_viewer_update_period_slider = self.server.add_gui_slider(
                "Update Period",
                min=1,
                max=100,
                step=1,
                initial_value=10,
                disabled=self.pause_training,
            )

            self.pause_training_button = self.server.add_gui_button("Pause Training")
            self.sh_order = self.server.add_gui_slider(
                "SH Order", min=1, max=4, step=1, initial_value=1
            )
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=1024
            )
            self.FoV_slider = self.server.add_gui_slider(
                    "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
                )
            self.near_plane_slider = self.server.add_gui_slider(
                "Near", min=0.1, max=30, step=0.5, initial_value=0.1
            )
            self.far_plane_slider = self.server.add_gui_slider(
                "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
            )

            self.show_train_camera = self.server.add_gui_checkbox(
                "Show Train Camera", initial_value=False
            )

            self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

            self.axis = self.server.add_frame("Axis", show_axes=False, axes_length=1000)

            self.time_bar = self.server.add_gui_slider(
                "Timestep", min=0, max=1000, step=1, initial_value=0, visible=False
            )

            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "render",
                ],
            )
            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        @self.renderer_output.on_update
        def _(_):
            self.need_update = True

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

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.toggle_axis.on_update
        def _(_):
            self.need_update = True
            self.axis.show_axes = self.toggle_axis.value

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
        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 20),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        self.debug_idx = 0
        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)
        
    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
        # position = -cam.R.T @ cam.T

        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()
        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        # breakpoint()
        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans
    
    def get_kwargs(self):
        out = {}
        if hasattr(self, "time_bar"):
            out["timestep"] = self.time_bar.value
        if hasattr(self, "mask_thresh"):
            out["mask_thresh"] = self.mask_thresh.value
        if hasattr(self, "invert_mask"):
            out["invert_mask"] = self.invert_mask.value

        return out

    def click_cb(self, pointer):
        if self.sam_enabled.value and self.add_sam_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos)

            self.add_points3d(self.camera, click_pos)
            print(self.points3d)

            self.viwer_need_update = True
        # elif self.draw_bbox.value:
        #     assert hasattr(pointer, "click_pos"), "please install our forked viser"
        #     click_pos = pointer.click_pos
        #     click_pos = torch.tensor(click_pos)
        #     cur_cam = self.camera
        #     if self.draw_flag:
        #         self.left_up.value = [
        #             int(cur_cam.image_width * click_pos[0]),
        #             int(cur_cam.image_height * click_pos[1]),
        #         ]
        #         self.draw_flag = False
        #     else:
        #         new_value = [
        #             int(cur_cam.image_width * click_pos[0]),
        #             int(cur_cam.image_height * click_pos[1]),
        #         ]
        #         if (self.left_up.value[0] < new_value[0]) and (
        #             self.left_up.value[1] < new_value[1]
        #         ):
        #             self.right_down.value = new_value
        #             self.draw_flag = True
        #         else:
        #             self.left_up.value = new_value
    
    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)

        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d)

    # no longer needed since can be extracted from langsam
    # def sam_encode_all_view(self):
    #     assert hasattr(self, "sam_predictor")
    #     self.sam_features = {}
    #     # NOTE: assuming all views have the same size
    #     for id, frame in self.origin_frames.items():
    #         # TODO: check frame dtype (float32 or uint8) and device
    #         self.sam_predictor.set_image(frame)
    #         self.sam_features[id] = self.sam_predictor.features

    @torch.no_grad()
    def update_sam_mask_with_point_prompt(
        self, points3d=None, save_mask=False, save_name="point_prompt_mask"
    ):
        points3d = points3d if points3d is not None else self.points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )
        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            assert len(points3d) > 0
            points2ds = project(cur_cam, points3d)
            img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)[
                "render"
            ]

            self.sam_predictor.set_image(
                np.asarray(to_pil_image(img.cpu())),
            )
            self.sam_features[idx] = self.sam_predictor.features
            # print(points2ds)
            mask, _, _ = self.sam_predictor.predict(
                point_coords=points2ds.cpu().numpy(),
                point_labels=np.array([1] * points2ds.shape[0], dtype=np.int64),
                box=None,
                multimask_output=False,
            )
            mask = torch.from_numpy(mask).to(torch.bool).cuda()
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask)

        weights /= weights_cnt + 1e-7

        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False

        if save_mask:
            for id, mask in enumerate(masks):
                mask = mask.cpu().numpy()[0, 0]
                img = Image.fromarray(mask)
                os.makedirs("tmp",exist_ok=True)
                img.save(f"./tmp/{save_name}-{id}.jpg")

        return masks, selected_mask

    @torch.no_grad()
    def sam_predict(self, image, cam):
        img = np.asarray(to_pil_image(image.cpu()))
        self.sam_predictor.set_image(img)
        if len(self.points3d) == 0:
            return
        _points2ds = project(cam, self.points3d)
        _mask, _, _ = self.sam_predictor.predict(
            point_coords=_points2ds.cpu().numpy(),
            point_labels=np.array([1] * _points2ds.shape[0], dtype=np.int64),
            box=None,
            multimask_output=False,
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).cuda()

        return _mask.squeeze(), _points2ds

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            if self.show_semantic_mask.value:
                out_img = output["semantic"][0].moveaxis(0, -1)
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if self.sam_enabled.value:
            if "sam_masks" in output and len(output["sam_masks"]) > 0:
                try:
                    out_img = torchvision.utils.draw_segmentation_masks(
                        out_img, output["sam_masks"][0]
                    )

                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        output["point2ds"][0][None, ...],
                        colors="blue",
                        radius=5,
                    )
                except Exception as e:
                    print(e)

        if (
            self.draw_bbox.value
            and self.draw_flag
            and (self.left_up.value[0] < self.right_down.value[0])
            and (self.left_up.value[1] < self.right_down.value[1])
        ):
            out_img[
                :,
                self.left_up.value[1] : self.right_down.value[1],
                self.left_up.value[0] : self.right_down.value[0],
            ] = 0

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)
    
    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])
        viser_cam = list(self.server.get_clients().values())[0].camera
        # viser_cam.up_direction = tf.SO3(viser_cam.wxyz) @ np.array([0.0, -1.0, 0.0])
        # viser_cam.look_at = viser_cam.position
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        # T = viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        # fovy = self.render_cameras[0].FoVy
        # fovx = self.render_cameras[0].FoVx
        # math.tan(self.render_cameras[0].FoVx / 2) / math.tan(self.render_cameras[0].FoVy / 2)
        # math.tan(fovx/2) / math.tan(fovy/2)

        # aspect = viser_cam.aspect
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)
    
    @torch.no_grad()
    def update(self):
        if self.need_update:
            times = []
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
                cam = MiniCam(W, H, camera.fov, camera.fov, znear, zfar, world_view_transform, full_proj_transform)
                # cam = self.camera
                # c2w = torch.from_numpy(get_c2w(camera)).to(self.device)
                try:
                    start = time.time()
                    out = render(
                        cam,
                        self.gaussian,
                        self.pipe,
                        self.background_tensor,
                    )
                    self.renderer_output.options = list(out.keys())
                    out = (
                        out[self.renderer_output.value]
                        .detach()
                        .cpu()
                        .clamp(min=0.0, max=1.0)
                        .numpy()
                        * 255.0
                    ).astype(np.uint8)
                    end = time.time()
                    times.append(end - start)
                except RuntimeError as e:
                    print(e)
                    continue
                out = np.moveaxis(out.squeeze(), 0, -1)
                client.set_background_image(out, format="jpeg")
                del out

            self.render_times.append(np.mean(times))
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"

    def render_loop(self):
        while True:
            try:
                self.update()
                time.sleep(0.001)
            except KeyboardInterrupt:
                return

class System:
    def __init__(self, gaussian, pipe, background_tensor):
        self.gaussian = gaussian
        self.pipe = pipe
        self.background_tensor = background_tensor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    args = parser.parse_args()
    # gaussians = GaussianModel(sh_degree=3)
    # gaussians.load_ply(os.path.join(args.model_path,
    #                                 "point_cloud",
    #                                 "iteration_" + str(args.iteration),
    #                                 "point_cloud.ply"))
    # gaussians.load_ply(args.gs_source)
    # parser = ArgumentParser(description="Training script parameters")
    # pipe = PipelineParams(parser)
    # background_tensor = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    # system = System(gaussians, pipe, background_tensor)
    viewer = ViserViewer(args)
    # viewer.set_system(system)
    while(True):
        viewer.update()
