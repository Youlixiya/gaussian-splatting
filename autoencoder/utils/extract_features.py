import os
import cv2
import numpy as np
from copy import deepcopy
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import alpha_clip
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tokenize_anything import model_registry, TapAutomaticMaskGenerator

class SAMImageDataset(Dataset):

    def __init__(self,
                 image_dir,
                 ):
        self.image_file_list = os.listdir(image_dir)
        self.image_path_list = [os.path.join(image_dir, i) for i in self.image_file_list]

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        feature_save_path = self.image_path_list[idx].replace('images', 'features').replace('jpg', 'pt')
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return image, feature_save_path


def collate_fn(batch):
    datas = []
    save_paths = []
    for data, save_path in batch:
        datas.append(data)
        save_paths.append(save_path)
    return datas, save_paths 
        

def parse_option():
    parser = argparse.ArgumentParser('argument for pre-processing')
    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--dataset_path', type=str, default="data/sam/images", help='root path of dataset')
    # parser.add_argument('--dataset_dir', type=str, required=True, help='dir of dataset')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--tap_type', type=str, default="tap_vit_l")
    parser.add_argument('--tap_ckpt', type=str, default="ckpts/tap/tap_vit_l_03f8ec.pkl")

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_option()

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    clip_model, preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="ckpts/alphaclip/clip_b16_grit+mim_fultune_4xe.pth", device=device) 
    mask_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224, 224)),
        transforms.Normalize(0.5, 0.26)
        ])
    # alpha = mask_transform(binary_mask * 255)
    # image_features = model.visual(image, alpha)
    tap = model_registry[args.tap_type](checkpoint=args.tap_ckpt).to(device)
    tap.text_decoder.reset_cache(max_batch_size=1024)
    tap_mask_generator = TapAutomaticMaskGenerator(tap)

    dataset = SAMImageDataset(args.dataset_path)
    sampler = DistributedSampler(dataset)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, sampler=sampler, drop_last=True)

    features_dir = args.dataset_path.replace('images', 'features')
    # test_image_paths = [os.path.join(test_image_dir, img_name) for img_name in os.listdir(test_image_dir)]
    # test_feature_dir = test_image_dir.replace('images', 'features')
    if args.local_rank == 0:
        os.makedirs(features_dir, exist_ok=True)
    # clip_features_sem_tokens = []
    for images, save_paths in tqdm(data_loader):
        
        for image, save_path in zip(images, save_paths):
            feature_save_path = save_path.split('.')
            with torch.no_grad():
            #   data = data.to(device)
                masks = tap_mask_generator.generate(image)
                image = preprocess(Image.fromarray(image)).half().unsqueeze(0).to(device)
                for i, mask in enumerate(masks):
                    torch.cuda.empty_cache()
                    alpha = mask_transform(Image.fromarray((mask['segmentation'] * 255).astype(np.uint8)))
                    # print(alpha.shape)
                    alpha = alpha.half().to(device).unsqueeze(dim=0)
                    clip_feature = clip_model.visual(image, alpha)
                    
                    cur_feature_save_path = deepcopy(feature_save_path)
                    cur_feature_save_path[0] = cur_feature_save_path[0] + f'_{i}'
                    cur_feature_save_path = '.'.join(cur_feature_save_path)
                    mask['clip_feature'] = torch.nn.functional.normalize(clip_feature, dim=-1).squeeze(0).cpu()
                    del clip_feature
                    # clip_features_sem_tokens.append({'clip_feature': clip_feature.cpu(), 'sem_token': mask['sem_token']})
                    torch.save(mask, cur_feature_save_path)
                    # torch.save({'clip_feature': clip_feature, 'sem_token': mask['sem_token']}, cur_feature_save_path)
    # torch.save(clip_features_sem_tokens, args.dataset_path.replace('images', 'clip_features_sem_tokens.pt'))      