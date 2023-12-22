import os
import json
import numpy as np
from PIL import Image

with open('data/Multiview-Segmentation-Data/fern (llff)/annotations.json', 'r') as f:
    annotations = json.load(f)
mask = np.zeros((756, 1008))
segmentation = annotations['annotations'][0]['segmentation'][0]
print(segmentation)
# print()
for i in range(len(segmentation)//2):
    w = int(segmentation[2* i])
    h = int(segmentation[2* i+1])
    print(w, h)
    mask[h, w] = 255
Image.fromarray(np.stack([mask.astype(np.uint8)]*3, -1)).save('mask.png')
# print(annotations['segmentation'])
