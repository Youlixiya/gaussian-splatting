
import os
import cv2
import torch
import torchvision
import random
import math
import clip
import numpy as np
from copy import deepcopy
from PIL import Image
import time
import viser
import viser.transforms as tf
import torch.nn.functional as F
from omegaconf import OmegaConf
from collections import deque
from typing import Any, Dict
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image, to_tensor
from scene.cameras import Simple_Camera, C2W_Camera, MiniCam
from gaussian_renderer import render
from scene.camera_scene import CamScene
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.camera_utils import project, unproject
from utils.extract_masks import MaskDataset
from utils.colormaps import ColormapOptions, apply_colormap
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, GaussianFeatureModel
from grounded_sam import GroundMobileSAM, GroundSAM

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
        self.gaussian = GaussianFeatureModel(sh_degree=3)
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.load_feature_params(cfg.feature_gs_source)
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)
        # self.set_system(System(self.gaussian, pipe, background_tensor))
        self.device = "cuda"
        self.port = 8080
        self.colors = np.random.random((500, 3))
        self.use_sam = False
        # self.guidance = None
        # self.stop_training = False
        # self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        
        self.points3d = []
         # front end related
        self.colmap_cameras = None
        self.render_cameras = None
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=-1, w=-1)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras
        #     self.maskdataset = MaskDataset(self.colmap_dir, self.colmap_cameras)
        #     print(self.maskdataset[0])

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.origin_frames = {}
        self.masks_2D = {}
        self.clip_model, self.preprocess = clip.load(cfg.clip_type, device=self.device)
        # self.text_segmentor = GroundMobileSAM(device=self.device)
        # self.text_segmentor = GroundSAM(device=self.device)
        # self.sam_predictor = self.text_segmentor.sam_predictor
        # self.sam_predictor.is_image_set = True
        # self.sam_features = {}
        with self.server.add_gui_folder("Sematic Query"):
            self.text_prompt = self.server.add_gui_text('Text Prompt', '')
            self.show_sematic_map = self.server.add_gui_checkbox("Show Sematic Map", initial_value=False)
            self.show_mask = self.server.add_gui_checkbox("Show Mask", initial_value=False)
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
                    "comp_rgb", 'masks'
                ],
            )
            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )
            self.extract_mesh_button = self.server.add_gui_button(
                "Extract Mesh"
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
    @torch.no_grad()
    def get_sematic_map(self, render_feature): 
        h, w = render_feature.shape[1:]
        text_prompt = clip.tokenize([self.text_prompt.value]).to(self.device)
        text_features = self.clip_model.encode_text(text_prompt)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100 * text_features @ torch.stack(self.gaussian.sematic_table.table, dim=-1).half()).softmax(dim=-1).squeeze(0)
        max_index = torch.argmax(similarity)
        print(max_index)
        query_embedding = self.gaussian.sematic_table.table[max_index]
        query_embedding_low_dim = self.gaussian.sematic_compressor(query_embedding.unsqueeze(0).float())
        sematic_map = F.cosine_similarity(query_embedding_low_dim, render_feature.reshape(self.gaussian.feature_dim, -1).permute(1, 0)).reshape(-1, h, w).permute(1, 2, 0)
        sematic_map = (sematic_map - sematic_map.min()) / (sematic_map.max() - sematic_map.min())
        # sematic_map = (sematic_map - np.min(sematic_map)) / (np.max(sematic_map) - np.min(sematic_map))
        # print(sematic_map)
        return sematic_map
    
    @torch.no_grad()
    def get_mask(self, render_feature):
        h, w = render_feature.shape[1:]
        instance_index = torch.argmax(self.gaussian.mask_decoder(render_feature.reshape(-1, h * w).permute(1, 0)).softmax(-1), dim=-1).cpu()
        print(instance_index)
        instance_map = self.gaussian.instance_colors[instance_index].reshape(h, w, 3)
        print(instance_map)
        return instance_map
    
    # @torch.no_grad()
    # def get_mask(self, render_feature):
    #     h, w = render_feature.shape[1:]
    #     instance_index = (torch.sigmoid(self.gaussian.mask_decoder(render_feature.reshape(-1, h * w).permute(1, 0)) + 0.5)).long().squeeze(-1).cpu()
    #     instance_map = self.gaussian.instance_colors[instance_index].reshape(h, w, 3)
    #     print(instance_map)
    #     return instance_map
    
    # @torch.no_grad()
    # def get_mask(self, render_feature):
    #     h, w = render_feature.shape[1:]
    #     text_prompt = clip.tokenize([self.text_prompt.value]).to(self.device)
    #     text_features = self.clip_model.encode_text(text_prompt)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     similarity = (100 * text_features @ torch.stack(self.gaussian.sematic_table.table, dim=-1).half()).softmax(dim=-1).squeeze(0)
    #     max_index = torch.argmax(similarity)
    #     query_embedding = self.gaussian.sematic_table.table[max_index]
    #     query_embedding_low_dim = self.gaussian.sematic_compressor(query_embedding.unsqueeze(0).float())
    #     # print(render_feature.shape)
    #     # print(query_embedding_low_dim.shape)
    #     pred_mask = torch.sigmoid(self.gaussian.mask_decoder(render_feature.unsqueeze(0) * query_embedding_low_dim[..., None, None])[0, 0, ...])
    #     # pred_mask = self.gaussian.mask_decoder(render_feature.unsqueeze(0) * query_embedding_low_dim[..., None, None])[0, 0, ...]
    #     return pred_mask
    
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
            
    def render(
        self,
        cam,
        local=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C
        if self.show_sematic_map.value and self.text_prompt.value:
            render_feature = render(cam, self.gaussian, self.pipe, self.background_tensor, render_feature=True)['render_feature']
            sematic_map = self.get_sematic_map(render_feature)
            heat_map = apply_colormap(sematic_map, ColormapOptions("turbo")).cpu().numpy().astype(np.float32)
            # print(heat_map.shape)
            # print(heat_map.shape)
            
            # image_np = (image.clone()[0].cpu().numpy())
            # sematic_map_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(sematic_map * 255), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            # sematic_map_rgb = np.float32(sematic_map_rgb) / 255

            # heatmap = sematic_map_rgb + image_np
            # heatmap = heatmap / np.max(heatmap)
            # heat_map = cv2.addWeighted (image_np, 0.7, sematic_map_rgb, 0.3, 0)
            # render_pkg['heat_map'] = (sematic_map_rgb / 255).astype(np.float32)
            Image.fromarray((heat_map * 255).astype(np.uint8)).save('1.jpg')
            render_pkg['heat_map'] = torch.tensor(heat_map, dtype=torch.float32)[None]
        
        if self.show_mask.value:
            render_feature = render(cam, self.gaussian, self.pipe, self.background_tensor, render_feature=True)['render_feature']
            pred_mask = self.get_mask(render_feature)
            Image.fromarray((pred_mask).cpu().numpy().astype(np.uint8)).save('1.jpg')
            render_pkg['mask_image'] = (pred_mask / 255).to(torch.float32)[None]
        
        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }
    
    @torch.no_grad()
    def update_mask(self, text_prompt) -> None:

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
            this_frame = render(
                cur_cam, self.gaussian, self.pipe, self.background_tensor
            )["render"]

            # breakpoint()
            # this_frame [c h w]
            # this_frame = this_frame.moveaxis(0, -1)[None, ...]
            mask = torch.FloatTensor(self.text_segmentor(to_pil_image(this_frame.cpu()), [f'{text_prompt}.'])[2])[None, ...].to(self.device)
            # print(mask.shape)
            if self.use_sam:
                print("Using SAM")
                self.sam_features[idx] = self.sam_predictor.features

            masks.append(mask)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

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
        return masks, selected_mask

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
            img_np = np.asarray(to_pil_image(img.cpu()))
            self.sam_predictor.set_image(
                img_np,
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
                # self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False
        if save_mask:
            for id, mask in enumerate(masks):
                mask = mask.cpu().numpy().transpose(1, 2, 0)
                mask_img = deepcopy(img_np)
                mask_img[mask] = (0.50 * mask_img[mask] + 0.50 * np.array([255, 0, 0])).astype(np.uint8)
                mask_img = Image.fromarray(mask_img)
                os.makedirs("tmp",exist_ok=True)
                mask_img.save(f"./tmp/{save_name}-{id}.jpg")

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
        if out_key in output.keys():
            out_img = output[out_key][0]  # H W C
        else:
            out_img = output['comp_rgb'][0]
        # if out_key == "comp_rgb":
        #     # if self.show_semantic_mask.value and "semantic" in output.keys():
        #     #     out_img = output["semantic"][0].moveaxis(0, -1)
        # elif out_key == "masks":
        #     out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        # if self.sam_enabled.value:
        #     if "sam_masks" in output and len(output["sam_masks"]) > 0:
        #         try:
        #             out_img = torchvision.utils.draw_segmentation_masks(
        #                 out_img, output["sam_masks"][0]
        #             )

        #             out_img = torchvision.utils.draw_keypoints(
        #                 out_img,
        #                 output["point2ds"][0][None, ...],
        #                 colors="blue",
        #                 radius=5,
        #             )
        #         except Exception as e:
        #             print(e)

        # if (
        #     self.draw_bbox.value
        #     and self.draw_flag
        #     and (self.left_up.value[0] < self.right_down.value[0])
        #     and (self.left_up.value[1] < self.right_down.value[1])
        # ):
        #     out_img[
        #         :,
        #         self.left_up.value[1] : self.right_down.value[1],
        #         self.left_up.value[0] : self.right_down.value[0],
        #     ] = 0

        self.renderer_output.options = list(output.keys())
        # if out_key in ['heat_map', 'mask_image']:
        #     return out_img
        # else:
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
                # camera = client.camera
                # W = self.resolution_slider.value
                # H = int(self.resolution_slider.value/camera.aspect)
                # znear = self.near_plane_slider.value
                # zfar = self.far_plane_slider.value
                # world_view_transform = torch.tensor(get_w2c(camera)).cuda()
                # world_view_transform[:3, [1, 2]] *= -1
                # world_view_transform = world_view_transform.transpose(0, 1)
                # projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=camera.fov, fovY=camera.fov).transpose(0,1).cuda()
                # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                # cam = MiniCam(W, H, camera.fov, camera.fov, znear, zfar, world_view_transform, full_proj_transform)
                cam = self.camera
                output = self.render(cam)
                out = self.prepare_output_image(output)
                # cam = self.camera
                # c2w = torch.from_numpy(get_c2w(camera)).to(self.device)
                # try:
                #     start = time.time()
                #     out = render(
                #         cam,
                #         self.gaussian,
                #         self.pipe,
                #         self.background_tensor,
                #     )
                #     self.renderer_output.options = list(out.keys())
                #     out = (
                #         out[self.renderer_output.value]
                #         .detach()
                #         .cpu()
                #         .clamp(min=0.0, max=1.0)
                #         .numpy()
                #         * 255.0
                #     ).astype(np.uint8)
                #     end = time.time()
                #     times.append(end - start)
                # except RuntimeError as e:
                #     print(e)
                #     continue
                # out = np.moveaxis(out.squeeze(), 0, -1)
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
    parser.add_argument("--feature_gs_source", type=str, required=True)  # feature gs pt file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    parser.add_argument("--clip_type", type=str, default='ViT-B/32')  # gs ply or obj file?
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