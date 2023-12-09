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

class AutoMaskExtractor:
    def __init__(self,
                 sam_checkpoint,
                 model_type,
                 data_path,
                 clip_type='ViT-B/32',
                 device='cuda') -> None:
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.clip_model, self.preprocess = clip.load(clip_type, device=device)
        downsample = os.path.basename(data_path).split('_')[-1]
        self.imgs_list = os.listdir(data_path)
        self.imgs_path_list = [os.path.join(data_path, img) for img in self.imgs_list]
        data_root = os.path.abspath(os.path.join(data_path, os.pardir))
        self.masks_path = os.path.join(data_root, f'masks_{downsample}')
        os.makedirs(self.masks_path, exist_ok=True)
        
    @torch.no_grad()
    def run(self):
        for i in trange(len(self.imgs_path_list)):
            img_path = self.imgs_path_list[i]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            save_name = self.imgs_list[i].split('.')[0] + '.npz'
            save_path = os.path.join(self.masks_path, save_name)
            masks = self.mask_generator.generate(image)
            masks_save = []
            for mask in masks:
                x, y, w, h = mask['bbox']
                img_roi = image[y : y + h, x : x + w, :]
                img_roi = Image.fromarray(img_roi)
                img_roi = self.preprocess(img_roi).unsqueeze(0).cuda()
                roifeat = self.clip_model.encode_image(img_roi).squeeze(0)
                roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
                masks_save.append({'mask': mask['segmentation'], 'clip_embedding': roifeat.cpu().half()})
            # torch.save(masks_save, save_path)
            np.savez_compressed(save_path, *masks_save)

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
        

class MaskDataset(Dataset):
    def __init__(self,
                 source_root,
                 cameras,
                 mask_dir='masks_4',
                 dino_features_dir = 'dino_features_4',
                 ):
        self.imgs_name = [camera.image_name for camera in cameras]
        self.masks_name = [img_name.split('.')[0] + '.npz' for img_name in self.imgs_name]
        self.masks_path = [os.path.join(source_root, mask_dir, mask_name) for mask_name in self.masks_name]
        self.dinov2_features_path = os.path.join(source_root, dino_features_dir, 'dinov2_features.npz')
        # self.masks = [self.load_mask_embedding(np.load(mask_path, allow_pickle=True)) for mask_path in self.masks_path]
    
    def __len__(self):
        return len(self.masks_path)
    
    def __getitem__(self, idx):
        # return self.masks[idx]
        return self.load_mask_embedding(idx)
    
    def load_dinov2_features(self):
        mask_embedding_array = np.load(self.dinov2_features_path, allow_pickle=True)
    
    def load_mask_embedding(self, idx):
        mask_embedding_array = np.load(self.masks_path[idx], allow_pickle=True)
        mask = []
        for i in range(len(mask_embedding_array)):
            array_name = 'arr_{}'.format(i)
            current_array = mask_embedding_array[array_name].tolist()
            mask.append(current_array)
        return mask
    # def parse_mask_embedding()
        

if __name__ == '__main__':
    # extractor = AutoMaskExtractor('Grounded-SAM/weights/sam_vit_h_4b8939.pth', 'vit_h', 
    #                               'data/360_v2/garden/images_4', )
    # extractor.run()
    extractor = DINOV2Extractor('data/360_v2/garden/images_4', )
    extractor.run()
    
            
            