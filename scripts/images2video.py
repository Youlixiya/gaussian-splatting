import os
import glob
import tqdm
import argparse
import imageio
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help="path to the images")
# parser.add_argument('--downscale', type=int, default=4)

opt = parser.parse_args()

images_name = os.listdir(opt.path)
images_name.sort()
images_path = [os.path.join(opt.path, image_name) for image_name in images_name]
print(images_name)
images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in images_path]
imageio.mimwrite(os.path.join(opt.path, '..', 'video.mp4'), images, fps=5, quality=8)
# for img_path in tqdm.tqdm(img_paths):
#     run_image(img_path)
