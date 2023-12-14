import os
from PIL import Image

root_path = 'data/lerf_data/figurines/images'
save_path = 'data/lerf_data/figurines/images_'
os.makedirs(save_path, exist_ok=True)
img_names = os.listdir(root_path)
img_paths = [os.path.join(root_path, img_name) for img_name in img_names]
save_paths = [os.path.join(save_path, img_name) for img_name in img_names]
imgs = [Image.open(img_path) for img_path in img_paths]
h, w = imgs[0].size[1], imgs[0].size[0]
for i, img in enumerate(imgs):
    img.resize((w, h)).save(save_paths[i])