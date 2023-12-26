
import os
import re
import cv2
import torch
import torchvision
import random
import math
import clip
import alpha_clip
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
from utils.color import generate_contrasting_colors
from utils.colormaps import ColormapOptions, apply_colormap
from utils.mask_utils import draw_mask
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, GaussianFeatureModel
from grounded_sam import GroundMobileSAM, GroundSAM
from lisa.lisa_pipeline import LISAPipeline

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
        self.cfg = cfg
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir
        self.gaussian = GaussianFeatureModel(sh_degree=3, gs_feature_dim=cfg.gs_feature_dim)
        self.gaussian.load_ply(self.gs_source)
        if cfg.feature_gs_source:
            self.gaussian.load_feature_params(cfg.feature_gs_source)
        parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(parser)
        self.device = "cuda:0"
        self.port = 8080
        self.colors = np.random.random((500, 3))
        self.renderer_output_cache = 'comp_rgb'
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.semantic_valid_num = None
        self.semantic_embeddings = None
        self.semantic_index = None
        self.semantic_colors = []
        self.instance_embeddings = []
        self.instance_colors = []
        self.colors = generate_contrasting_colors(500)
        self.point_pos = None
        # self.points3d = []
         # front end related
        self.colmap_cameras = None
        self.render_cameras = None
        if self.colmap_dir is not None:
            scene = CamScene(self.colmap_dir, h=-1, w=-1)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras
            img_suffix = os.listdir(os.path.join(self.colmap_dir, cfg.images))[0].split('.')[-1]
            self.imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in self.colmap_cameras]
            self.imgs_path = [os.path.join(self.colmap_dir, cfg.images, img_name) for img_name in self.imgs_name]
        #     self.maskdataset = MaskDataset(self.colmap_dir, self.colmap_cameras)
        #     print(self.maskdataset[0])

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device=self.device
        )
        self.origin_frames = {}
        self.masks_2D = {}
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
        #         cfg.clip_model_type,  # e.g., ViT-B-16
        #         pretrained=cfg.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
        #         precision="fp16",
        #     )
        # self.clip_model = self.clip_model.to(self.device)
        # self.tokenizer = open_clip.get_tokenizer(cfg.clip_model_type)
        #load clip
        self.clip_model, self.preprocess = clip.load(cfg.clip_model_type, mask_prompt_depth=3, device=self.device)
        clip_ckpt = torch.load(cfg.clip_model_pretrained)
        self.clip_model.load_state_dict(clip_ckpt)
        neg_features = self.clip_model.encode_text(clip.tokenize(['background']).to(self.device))
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
        self.neg_features = neg_features
        #load llava
        if cfg.lisa:
            self.vlm_name = 'LISA'
            self.lisa_pipeline = LISAPipeline(cfg.lisa_model_type, local_rank=1, load_in_4bit=False, load_in_8bit=True, conv_type=cfg.lisa_conv_type)
        else:
            self.vlm_name = ''
        self.lisa_mask = None
        # text_prompt = alpha_clip.tokenize([self.text_prompt.value]).to(self.device)
        
        # self.clip_model, self.preprocess = alpha_clip.load(cfg.clip_model_type, device=self.device, alpha_vision_ckpt_pth=cfg.clip_model_pretrained, lora_adapt=False, rank=-1)
        
        os.makedirs('tmp', exist_ok=True)
        self.instance_feature_bg_color = torch.tensor([0] * self.gaussian.instance_feature_dim, dtype=torch.float32, device=self.device)
        # self.clip_model, self.preprocess = clip.load(cfg.clip_type, device=self.device)
        # self.text_segmentor = GroundMobileSAM(device=self.device)
        # self.text_segmentor = GroundSAM(device=self.device)
        # self.sam_predictor = self.text_segmentor.sam_predictor
        # self.sam_predictor.is_image_set = True
        # self.sam_features = {}
        with self.server.add_gui_folder("semantic Query"):
            self.text_prompt = self.server.add_gui_text('Text Prompt', '')
            self.vlm_decoder = self.server.add_gui_checkbox(f'{self.vlm_name} Decoder', initial_value=False, visible=cfg.lisa)
            self.vlm_chat = self.server.add_gui_button(f"{self.vlm_name} Chat", visible=cfg.lisa)
            self.vlm_output = self.server.add_gui_text(f'{self.vlm_name} Output', '', visible=cfg.lisa)
            self.show_semantic_map = self.server.add_gui_checkbox("Show semantic Map", initial_value=False)
            self.show_instance_map = self.server.add_gui_checkbox("Show Instance Map", initial_value=False)
            self.show_instance_mask = self.server.add_gui_checkbox("Show Instance Mask", initial_value=False)
            self.show_semantic_mask = self.server.add_gui_checkbox("Show semantic Mask", initial_value=False)
            self.mask_threshold = self.server.add_gui_slider('Mask Threshold', min=0, max=1, step=0.01, initial_value=0.5)
            self.text_query_threshold = self.server.add_gui_slider('Text Query Threshold', min=0, max=1, step=0.01, initial_value=0.8)
            self.mask_colors_picker = self.server.add_gui_rgb("Mask Colors Picker", initial_value=[255, 0, 0])
            self.add_instance_embedding = self.server.add_gui_checkbox('Add Instance embedding', False)
            self.selected_instance_num = self.server.add_gui_text('Selected Instance Num', '0')
            self.selected_semantic_num = self.server.add_gui_text('Selected Semantic Num', '0')
            # self.semantic_topk = self.server.add_gui_slider('Semantic Top K', min=1, max=len(self.gaussian.instance_embeddings), step=1, initial_value=1)
            self.add_semantic_embedding = self.server.add_gui_button("Add Semantic Embedding")
            self.clear_instance_embedding = self.server.add_gui_button("Clear Instance Embedding")
            self.clear_semantic_embedding = self.server.add_gui_button("Clear Semantic Embedding")
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

        @self.text_prompt.on_update
        def _(_):
            self.need_update = True
        
        @self.vlm_decoder.on_update
        def _(_):
            self.need_update = True
        
        @self.vlm_chat.on_click
        def _(_):
            self.get_mask_by_lisa()
            # if self.cfg.llava:
            #     self.get_object_name_from_implicit_query()
            # else:
            #     self.get_object_name_by_gemini()
            self.need_update = True
        
        @self.show_semantic_map.on_update
        def _(_):
            self.need_update = True
            
        @self.show_instance_map.on_update
        def _(_):
            self.need_update = True
            
        @self.show_instance_mask.on_update
        def _(_):
            self.need_update = True
        
        @self.show_semantic_mask.on_update
        def _(_):
            self.need_update = True
        
        @self.mask_threshold.on_update
        def _(_):
            self.need_update = True
        
        @self.text_query_threshold.on_update
        def _(_):
            self.select_semantic_embeddings()
            self.need_update = True
        
        @self.mask_colors_picker.on_update
        def _(_):
            self.need_update = True
            # print(self.mask_colors_picker.value)
        
        # @self.semantic_topk.on_update
        # def _(_):
        #     self.need_update = True
        
        @self.add_semantic_embedding.on_click
        def _(_):
            self.select_semantic_embeddings()
            self.need_update = True
        
        @self.clear_instance_embedding.on_click
        def _(_):
            self.instance_embeddings = []
            self.instance_colors = []
            self.selected_instance_num.value = '0'
        
        @self.clear_semantic_embedding.on_click
        def _(_):
            self.semantic_embeddings = None
            self.semantic_index = None
            # self.object_name = None
            self.semantic_colors = None
            self.semantic_valid_num = None
            self.selected_semantic_num.value = '0'
        
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
            self.frustums = []
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
            self.need_update=True
            
    @torch.no_grad()
    def select_semantic_embeddings(self):
        
        text_prompt = clip.tokenize(self.text_prompt.value.split(',')).to(self.device)
        # text_prompt = alpha_clip.tokenize([self.text_prompt.value]).to(self.device)
        pos_features = self.clip_model.encode_text(text_prompt)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        
        total_features = torch.cat([self.neg_features, pos_features])
        total_mm = self.gaussian.clip_embeddings @ total_features.T
        pos_mm = total_mm[:, 1:]
        neg_mm = total_mm[:, [0]].repeat(1, pos_mm.shape[-1])
        # print(pos_mm.shape)
        # print(pos_mm.shape)
        total_similarity = torch.stack([pos_mm, neg_mm], dim=-1)
        softmax = (100 * total_similarity).softmax(-1)
        pos_softmax = softmax[..., 0]
        valid_mask = pos_softmax > self.text_query_threshold.value
        self.semantic_valid_num = valid_mask.sum(0)
        semantic_embeddings = []
        for i in range(valid_mask.shape[-1]):
            semantic_embeddings.append(self.gaussian.instance_embeddings[valid_mask[:, i], :])
        self.semantic_embeddings = torch.cat(semantic_embeddings)
        # self.semantic_embeddings = self.gaussian.instance_embeddings[torch.nonzero(valid_mask.float()).reshape(-1), :]
        self.selected_semantic_num.value = str(len(self.semantic_valid_num))
        
        
        # pos_softmax = softmax[..., 0].permute(1, 0)
        # self.semantic_index = torch.topk(pos_softmax, k=int(self.semantic_topk.value), dim=-1)[1]
        # # # print(similarity)
        # # valid_index = similarity > self.text_query_threshold.value
        # # self.semantic_embeddings = self.gaussian.instance_embeddings[valid_index, :]
        # self.selected_semantic_num.value = str(len(self.semantic_index))
        # semantic_colors = torch.rand((len(self.semantic_index), 3), device=self.device)
        # self.semantic_colors = torch.stack([semantic_colors] * int(self.semantic_topk.value), dim=-1).reshape(-1, 3)
    
    @torch.no_grad()
    def get_mask_by_lisa(self):
        if self.text_prompt.value:
            result_list, mask_result_list, mask_list, mask_rgb_list, output_str = self.lisa_pipeline(self.text_prompt.value, image=self.image)
            self.lisa_mask = mask_list[0]
            self.vlm_output.value = output_str
        else:
            self.vlm_output.value = 'Please provide text prompt!'
        print(self.vlm_output.value)
    
    @torch.no_grad()
    def get_semantic_map(self, render_feature):
        h, w = render_feature.shape[1:]
        similarity_map = (F.normalize(render_feature.reshape(-1, h * w), dim=0).permute(1, 0) @ F.normalize(self.semantic_embeddings, dim=1).T).reshape(h, w, -1)
        return similarity_map
    
    @torch.no_grad()
    def get_instance_mask(self, render_feature):
        h, w = render_feature.shape[1:]
        # instance_index = torch.argmax((self.gaussian.instance_embeddings @ self.gaussian.instance_decoder(render_feature.reshape(-1, h * w))[self.gaussian.feature_dim:, :]).T.softmax(-1), dim=-1).cpu()
        instance_index = torch.argmax((F.normalize(render_feature.reshape(-1, h * w).permute(1, 0), dim=1) @ self.gaussian.instance_embeddings.T).softmax(-1), dim=-1).cpu()
        # print(instance_index)
        instance_map = self.gaussian.instance_colors[instance_index].reshape(h, w, 3)
        return instance_map

    # @torch.no_grad()
    # def get_semantic_mask(self, render_feature):
    #     h, w = render_feature.shape[1:]
    #     # semantic_embeddings_low_dim = self.gaussian.semantic_compressor(self.gaussian.semantic_embeddings.float())
    #     semantic_index = torch.argmax((render_feature.reshape(-1, h * w).permute(1, 0) @ self.gaussian.semantic_embeddings.T).softmax(-1), dim=-1).cpu()
    #     # semantic_index = torch.argmax((self.gaussian.semantic_decoder(render_feature.reshape(-1, h * w).permute(1, 0)) @ semantic_embeddings_low_dim.T).softmax(-1), dim=-1).cpu()
    #     # print(instance_index)
    #     semantic_map = self.gaussian.semantic_colors[semantic_index].reshape(h, w, 3)
    #     return semantic_map
    
    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]
        # image = cv2.cvtColor(cv2.imread(self.imgs_path[idx]), cv2.COLOR_BGR2RGB)
        # H, W = image.shape[:2]
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
        # frustum = self.server.add_camera_frustum(
        #     f"/colmap/frame_{idx}/frustum",
        #     fov=cam.FoVy,
        #     aspect=W / H,
        #     scale=0.15,
        #     image=image, # 0-255 uint8 H W C
        #     visible=False,
        # )
        self.frames.append(frame)
        # self.frustums.append(frustum)

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
        # def attach_callback(
        #     frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        # ) -> None:
        #     @frustum.on_click
        #     def _(_) -> None:
        #         for client in self.server.get_clients().values():
        #             client.camera.wxyz = frame.wxyz
        #             client.camera.position = frame.position

        # attach_callback(frustum, frame)
    
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
        # h, w = render_feature.shape[1:]
        if self.show_instance_map.value and self.point_pos is None:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos).cuda()
            self.point_pos = click_pos

            self.viwer_need_update = True
    
    def set_system(self, system):
        self.system = system

    # def clear_points3d(self):
    #     self.points3d = []

    # def add_points3d(self, camera, points2d, update_mask=False):
    #     depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
    #         "depth_3dgs"
    #     ]
    #     unprojected_points3d = unproject(camera, points2d, depth)
    #     self.points3d += unprojected_points3d.unbind(0)

    #     if update_mask:
    #         self.update_sam_mask_with_point_prompt(self.points3d)
    @torch.no_grad()
    def render(
        self,
        cam,
        local=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image = render_pkg['render']
        render_pkg.pop('viewspace_points')
        render_pkg.pop('visibility_filter')
        render_pkg.pop('radii')
        # image, viewspace_point_tensor, _, radii = (
        #     render_pkg["render"],
        #     render_pkg["viewspace_points"],
        #     render_pkg["visibility_filter"],
        #     render_pkg["radii"],
        # )
        
        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        self.image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))
        self.image.save('tmp/rgb.jpg')
        render_pkg["comp_rgb"] = image  # 1 H W C
        semantic_rendered_feature = None
        if self.show_instance_map or self.show_instance_map or self.show_instance_mask or self.show_semantic_mask:
            # render_feature = render(cam, self.gaussian, self.pipe, self.background_tensor, render_feature=True)['render_feature']
            rendered_feature = render(cam, self.gaussian, self.pipe, self.instance_feature_bg_color, render_feature=True, override_feature=self.gaussian.instance_features)['render_feature']
            h, w = rendered_feature.shape[1:]
            instance_feature = F.normalize(rendered_feature.reshape(-1, h * w), dim=0).reshape(-1, h, w)
            # total_rendered_feature = [rendered_feature]
            # if self.gaussian.rgb_decode:
            #     total_rendered_feature.append(render_pkg['render'])
            # if self.gaussian.depth_decode:
            #     total_rendered_feature.append(render_pkg['depth_3dgs'])
            # total_rendered_feature = torch.cat(total_rendered_feature, dim=0)
            # total_rendered_feature = total_rendered_feature.reshape(-1, h*w).permute(1, 0)
            # if self.gaussian.feature_aggregator:
            #     total_rendered_feature = F.normalize(self.gaussian.feature_aggregator(total_rendered_feature), dim=-1)
            # else:
            #     total_rendered_feature = F.normalize(total_rendered_feature, dim=-1)
            # total_rendered_feature = total_rendered_feature.permute(1, 0).reshape(-1, h, w)
            render_pkg['rendered_feature'] = apply_colormap(rendered_feature.permute(1, 2, 0), ColormapOptions(colormap="pca"))[None]
            render_pkg['instance_feature'] = apply_colormap(instance_feature.permute(1, 2, 0), ColormapOptions(colormap="pca"))[None]
            Image.fromarray((render_pkg['rendered_feature'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/rendered_feature.jpg')
            Image.fromarray((render_pkg['instance_feature'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/instance_feature.jpg')
            
        else:
            rendered_feature = None
        if self.show_semantic_map.value and self.semantic_embeddings is not None:
            # if semantic_rendered_feature is None:
            #     semantic_rendered_feature = render(cam, self.gaussian, self.pipe, self.semantic_feature_bg_color, render_feature=True, override_feature=self.gaussian.semantic_features)['render_feature']
            similarity_map = self.get_semantic_map(instance_feature)
            # print(similarity_map)
            masks = similarity_map > self.mask_threshold.value
            masks_all = masks.any(-1)
            semantic_mask_map = image.clone()[0]
            sematic_object_map = image.clone()[0]
            start_index = 0
            for i in range(len(self.semantic_valid_num)):
                mask = masks[..., start_index:start_index + self.semantic_valid_num[i]].any(-1)
                semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + torch.tensor(self.colors[i], device=self.device) / 255 * 0.5
                # semantic_mask_map[~mask, :] = semantic_mask_map[~mask, :] * 0.5 + torch.tensor([0, 0, 0], device=self.device) * 0.5
                start_index += self.semantic_valid_num[i]
            # for i, mask in enumerate(masks.permute(2, 0, 1)):
            #     semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + torch.tensor(self.colors[i], device=self.device) / 255 * 0.5
            # semantic_rgb_map[mask, :] = semantic_map[mask, :]
            # semantic_rgb_map[mask, :] = semantic_rgb_map[mask, :] * 0.5 + (semantic_color / 255) * 0.5
            # semantic_mask_map[mask, :] = semantic_mask_map[mask, :] * 0.5 + torch.tensor(self.mask_colors_picker.value, dtype=torch.float32, device=self.device) / 255 * 0.5
            sematic_object_map[~masks_all, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
            render_pkg['semantic_mask_map'] = semantic_mask_map[None]
            render_pkg['sematic_object_map'] = sematic_object_map[None]
            # Image.fromarray((render_pkg['semantic_heat_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/semantic_heat_map.jpg')
            Image.fromarray((render_pkg['semantic_mask_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/semantic_mask_map.jpg')
            Image.fromarray((render_pkg['sematic_object_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/sematic_object_map.jpg')
        
        if self.show_instance_map.value:
            # if self.point_pos is not None:
            #     if self.instance_feature is None:
            if (self.add_instance_embedding.value and self.point_pos is not None) or self.lisa_mask is not None:
                if self.lisa_mask is not None:
                    lisa_mask = torch.tensor(self.lisa_mask, torch.bool, device=self.device)
                    mask_instance_feature = instance_feature[:, lisa_mask].permute(1, 0)
                    instance_embeddings_index = torch.argmax(mask_instance_feature @ self.gaussian.instance_embeddings.T, dim=-1)
                    unique_index, counts = torch.unique(instance_embeddings_index)
                    instance_feature = self.gaussian.instance_embeddings[unique_index[torch.argmax(counts)]]
                    self.lisa_mask = None
                    
                else:
                    instance_feature = instance_feature[:, (self.point_pos[1] * h).long(), (self.point_pos[0] * w).long()][None]
                    instance_embedding_index = torch.argmax((instance_feature @ self.gaussian.instance_embeddings.T).softmax(-1))
                    instance_feature = self.gaussian.instance_embeddings[instance_embedding_index]
                    # if instance_feature not in self.instance_features:
                    self.point_pos = None
                self.instance_embeddings.append(instance_feature)
                self.instance_colors.append(self.mask_colors_picker.value)
                self.selected_instance_num.value = str(len(self.instance_embeddings))
                # print(self.point_pos)
                # print(self.render_feature)
            if self.instance_embeddings:
                similarity_map = (instance_feature.reshape(-1, h * w).permute(1, 0) @ torch.stack(self.instance_embeddings)).reshape(h, w, -1)
                # similarity_map = F.cosine_similarity(rendered_feature.reshape(-1, h*w).permute(1, 0), torch.stack(self.instance_features)).reshape(h, w, -1)
                # instance_map = apply_colormap(similarity_map, ColormapOptions(colormap="turbo", normalize=True, colormap_min=-1, colormap_max=1))
                masks = (similarity_map > self.mask_threshold.value)
                masks_all_instance = masks.any(-1)
                # instance_mask_map = (image.clone()[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                instance_mask_map = image.clone()[0]
                instance_object_map = image.clone()[0]
                # instance_mask_map[mask, :] = instance_map[mask, :]
                for i, mask in enumerate(masks.permute(2, 0, 1)):
                    instance_mask_map[mask, :] = instance_mask_map[mask, :] * 0.5 + torch.tensor(self.colors[i], dtype=torch.float32, device=self.device) /255 * 0.5
                # draw_mask(instance_mask_map, masks_all_instance.cpu().numpy())
                # instance_mask_map = torch.tensor(instance_mask_map, dtype=torch.float32)
                instance_mask_map[~masks_all_instance, :] /= 2
                instance_object_map[~masks_all_instance, :] = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                # render_pkg['instance_heat_map'] = instance_map[None]
                render_pkg['instance_mask_map'] = instance_mask_map[None]
                render_pkg['instance_object_map'] = instance_object_map[None]
                # Image.fromarray((render_pkg['instance_heat_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/instance_heat_map.jpg')
                Image.fromarray((render_pkg['instance_mask_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/instance_mask_map.jpg')
                Image.fromarray((render_pkg['instance_object_map'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/instance_object_map.jpg')
        
        if self.show_instance_mask.value:
            pred_instance_mask = self.get_instance_mask(instance_feature)
            # Image.fromarray((pred_instance_mask).cpu().numpy().astype(np.uint8)).save('1.jpg')
            render_pkg['instance_masks'] = (pred_instance_mask / 255).to(torch.float32)[None]
            # instance_feature = self.gaussian.instance_decoder(render_feature.reshape(-1, h*w).permute(1, 0)).reshape(h, w, -1)
            Image.fromarray((render_pkg['instance_masks'][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/instance_masks.jpg')
        
        depth = render_pkg['depth_3dgs']
        depth = depth.permute(1, 2, 0)[None].repeat(1, 1, 1, 3)
        render_pkg["depth"] = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
        Image.fromarray((render_pkg["depth"][0].cpu().numpy() * 255).astype(np.uint8)).save('tmp/depth.jpg')
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)
        self.need_update = False
        return {
            **render_pkg,
        }
    
    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        if out_key not in output.keys():
            out_img = output['comp_rgb'][0]
        else:
            out_img = output[out_key]  # H W C
            if out_img is None:
                out_img = output['comp_rgb'][0]
                self.renderer_output.value = self.renderer_output_cache
            else:
                out_img = out_img[0]
                self.renderer_output_cache = out_key
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

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
                cam = self.camera
                output = self.render(cam)
                out = self.prepare_output_image(output)
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
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--feature_gs_source", type=str, default='')  # feature gs pt file? 
    parser.add_argument("--gs_feature_dim", type=int, default=16)
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    parser.add_argument("--clip_model_type", type=str, default='ViT-B/16')  # gs ply or obj file?
    parser.add_argument("--clip_model_pretrained", type=str, default='mask_adapted_clip.pt')
    parser.add_argument("--lisa_model_type", type=str, default='xinlai/LISA-7B-v1')
    parser.add_argument("--lisa_conv_type", type=str, default='conv_type')
    parser.add_argument("--lisa", action='store_true')
    # parser.add_argument("--clip_model_pretrained", type=str, default='clip_b16_grit+mim_fultune_4xe.pth')
    args = parser.parse_args()
    viewer = ViserViewer(args)
    while(True):
        viewer.update()