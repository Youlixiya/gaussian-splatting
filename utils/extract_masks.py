import os
from typing import Any
import cv2
import clip
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch.utils.data import Dataset

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

class MaskDataset(Dataset):
    def __init__(self,
                 source_root,
                 cameras,
                 mask_dir='masks_4'):
        self.imgs_name = [camera.image_name for camera in cameras]
        self.masks_name = [img_name.split('.')[0] + '.npz' for img_name in self.imgs_name]
        self.masks_path = [os.path.join(source_root, mask_dir, mask_name) for mask_name in self.masks_name]
        # self.masks = [self.load_mask_embedding(np.load(mask_path, allow_pickle=True)) for mask_path in self.masks_path]
    
    def __len__(self):
        return len(self.masks_path)
    
    def __getitem__(self, idx):
        # return self.masks[idx]
        return self.load_mask_embedding(idx)
    
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
    extractor = AutoMaskExtractor('Grounded-SAM/weights/sam_vit_h_4b8939.pth', 'vit_h', 
                                  'data/360_v2/garden/images_4', )
    extractor.run()
    
            
            