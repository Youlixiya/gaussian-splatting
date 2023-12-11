import os
from typing import Any
import cv2
import clip
import torch
import math
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as T
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import Dataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# class AutoMaskExtractor:
#     def __init__(self,
#                  sam_checkpoint,
#                  model_type,
#                  data_path,
#                  clip_type='ViT-B/32',
#                  device='cuda') -> None:
#         sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         self.mask_generator = SamAutomaticMaskGenerator(sam)
#         self.clip_model, self.preprocess = clip.load(clip_type, device=device)
#         downsample = os.path.basename(data_path).split('_')[-1]
#         self.imgs_list = os.listdir(data_path)
#         self.imgs_path_list = [os.path.join(data_path, img) for img in self.imgs_list]
#         data_root = os.path.abspath(os.path.join(data_path, os.pardir))
#         self.masks_path = os.path.join(data_root, f'masks_{downsample}')
#         os.makedirs(self.masks_path, exist_ok=True)
        
#     @torch.no_grad()
#     def run(self):
#         for i in trange(len(self.imgs_path_list)):
#             img_path = self.imgs_path_list[i]
#             image = cv2.imread(img_path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             save_name = self.imgs_list[i].split('.')[0] + '.npz'
#             save_path = os.path.join(self.masks_path, save_name)
#             masks = self.mask_generator.generate(image)
#             masks_save = []
#             for mask in masks:
#                 x, y, w, h = mask['bbox']
#                 img_roi = image[y : y + h, x : x + w, :]
#                 img_roi = Image.fromarray(img_roi)
#                 img_roi = self.preprocess(img_roi).unsqueeze(0).cuda()
#                 roifeat = self.clip_model.encode_image(img_roi).squeeze(0)
#                 roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
#                 masks_save.append({'mask': mask['segmentation'], 'clip_embedding': roifeat.cpu().half()})
#             # torch.save(masks_save, save_path)
#             np.savez_compressed(save_path, *masks_save)

class AutoEmbedidngExtractor:
    def __init__(self,
                 sam_checkpoint,
                 model_type,
                 data_path,
                 clip_type='ViT-B/32',
                 clip_embedding_dim=512,
                 device='cuda') -> None:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.clip_model, self.preprocess = clip.load(clip_type, device=device)
        self.clip_embedding_dim = clip_embedding_dim
        self.device = device
        downsample = os.path.basename(data_path).split('_')[-1]
        self.imgs_list = os.listdir(data_path)
        self.imgs_path_list = [os.path.join(data_path, img) for img in self.imgs_list]
        data_root = os.path.abspath(os.path.join(data_path, os.pardir))
        self.embeddings_path = os.path.join(data_root, f'embeddings_{downsample}')
        os.makedirs(self.embeddings_path, exist_ok=True)
        
    @torch.no_grad()
    def run(self):
        for i in trange(len(self.imgs_path_list)):
            img_path = self.imgs_path_list[i]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            save_name = self.imgs_list[i].split('.')[0] + '.npz'
            save_path = os.path.join(self.embeddings_path, save_name)
            masks = self.mask_generator.generate(image)
            # masks_save = []
            embedding_map = torch.zeros((h, w, self.clip_embedding_dim), dtype=torch.float16, device=self.self.device)
            for mask in masks:
                x, y, w, h = mask['bbox']
                img_roi = image[y : y + h, x : x + w, :]
                img_roi = Image.fromarray(img_roi)
                img_roi = self.preprocess(img_roi).unsqueeze(0).cuda()
                roifeat = self.clip_model.encode_image(img_roi).squeeze(0)
                roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
                embedding_map[torch.tenor(mask, dtype=torch.bool), :] += roifeat
                # masks_save.append({'mask': mask['segmentation'], 'clip_embedding': roifeat.cpu().half()})
            embedding_map = torch.nn.functional.normalize(embedding_map, dim=-1)
            embeddings = np.unique(embedding_map.cpu().numpy().reshape(-1, self.clip_embedding_dim), axis=0)
            
            # torch.save(masks_save, save_path)
            np.savez_compressed(save_path, {'embedding_map':embedding_map, 'embeddings':embeddings})

class DINOV2Extractor:
    def __init__(self,
                 data_path,
                 img_size=896,
                 dino_type='dinov2_vitb14',
                 device='cuda',
                 pca_dim=16):
        downsample = os.path.basename(data_path).split('_')[-1]
        self.imgs_list = os.listdir(data_path)
        self.imgs_path_list = [os.path.join(data_path, img) for img in self.imgs_list]
        data_root = os.path.abspath(os.path.join(data_path, os.pardir))
        self.dino_features_path = os.path.join(data_root, f'dino_features_{downsample}')
        self.dino_features_save_path = os.path.join(self.dino_features_path, 'dinov2_features.npz')
        os.makedirs(self.dino_features_path, exist_ok=True)
        self.trans = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        self.device = device
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', dino_type).to(device)
        self.dinov2.eval()
        self.pca_dim = pca_dim
        self.pca = PCA(n_components=pca_dim)
    @torch.no_grad()
    def run(self):
        dinov2_features = []
        for i in trange(len(self.imgs_path_list)):
            img_path = self.imgs_path_list[i]
            image = self.trans(Image.open(img_path))[None, ...].to(self.device)
            features = self.dinov2.forward_features(image)['x_norm_patchtokens'].squeeze(0)
            feature_num, feature_dim = features.shape
            h = w = int(math.sqrt(feature_num))
            features_pca = self.pca.fit_transform(features.reshape(-1, feature_dim).cpu().numpy())
            features_pca = features_pca.reshape(h, w, self.pca_dim).transpose(2, 0, 1)
            dinov2_features.append(torch.Tensor(features_pca))
            # torch.save(masks_save, save_path)
        np.savez_compressed(self.dino_features_save_path, *dinov2_features)
        

# class MaskDataset(Dataset):
#     def __init__(self,
#                  source_root,
#                  cameras,
#                  mask_dir='masks_4',
#                  ):
#         self.imgs_name = [camera.image_name for camera in cameras]
#         self.masks_name = [img_name.split('.')[0] + '.npz' for img_name in self.imgs_name]
#         self.masks_path = [os.path.join(source_root, mask_dir, mask_name) for mask_name in self.masks_name]
    
#     def __len__(self):
#         return len(self.masks_path)
    
#     def __getitem__(self, idx):
#         # return self.masks[idx]
#         return self.load_mask_embedding(idx)
    
#     def load_dinov2_features(self):
#         mask_embedding_array = np.load(self.dinov2_features_path, allow_pickle=True)
    
#     def load_mask_embedding(self, idx):
#         mask_embedding_array = np.load(self.masks_path[idx], allow_pickle=True)
#         mask = []
#         for i in range(len(mask_embedding_array)):
#             array_name = 'arr_{}'.format(i)
#             current_array = mask_embedding_array[array_name].tolist()
#             mask.append(current_array)
#         return mask
    
class MaskDataset(Dataset):
    def __init__(self,
                 source_root,
                 cameras,
                 mask_dir='instance_4/Annotations',
                 device='cuda'
                 ):
        self.imgs_name = [camera.image_name for camera in cameras]
        self.masks_name = [img_name.split('.')[0] + '.png' for img_name in self.imgs_name]
        self.masks_path = [os.path.join(source_root, mask_dir, mask_name) for mask_name in self.masks_name]
        self.masks = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB) for mask_path in self.masks_path], axis=0)).to(device)
        self.device = device
        self.count_instance_num()
        self.asign_label()
        del self.masks
        self.masks = None
        # print(torch.sum(self.mask_labels))
        torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.mask_labels)
    
    def __getitem__(self, idx):
        # return self.masks[idx]
        return self.mask_labels[idx]
    
    @torch.no_grad()
    def count_instance_num(self):
        self.unique_colors = []
        for mask in tqdm(self.masks):
            self.unique_colors += torch.unique(mask.reshape(-1, 3), dim=0).tolist()
        self.unique_colors = torch.unique(torch.tensor(self.unique_colors, dtype=torch.uint8), dim=0)
        
    @torch.no_grad()
    def asign_label(self):
        n, h, w, c = self.masks.shape
        self.mask_labels = torch.zeros((n, h, w, 1), device=self.device)
        for i in trange(len(self.unique_colors)):
            mask = self.masks.reshape(n * h * w, -1)
            # print(self.unique_colors[i])
            target_color = self.unique_colors[i][None, :].to(self.device).expand(mask.size())
            index = torch.eq(mask, target_color).all(dim=1).reshape(n, h, w)
            # print(torch.sum(index.float()))
            # index = (self.masks.reshape(n * h * w, -1) == self.unique_colors[i].to(self.device)).reshape(n, h, w)
            self.mask_labels[index, :] = i
        self.mask_labels = self.mask_labels.long()
        # print(torch.sum(self.mask_labels))
            
            
    
   
    
    
        

if __name__ == '__main__':
    extractor = AutoEmbeddingExtractor('Grounded-SAM/weights/sam_vit_h_4b8939.pth', 'vit_h', 
                                  'data/360_v2/garden/images_4', )
    extractor.run()
    # extractor = DINOV2Extractor('data/360_v2/garden/images_4', )
    # extractor.run()
    
            
            