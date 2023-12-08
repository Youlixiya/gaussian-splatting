import numpy as np
import torch
# import torch

# mask_embedding = torch.load('data/360_v2/garden/masks_4/DSC07956.pt')
# mask_embedding = [for mask in mask_embedding]
# np.savez_compressed('bool_arrays.npz', *mask_embedding)
# torch.save(mask_embedding, '1.pth', _use_new_zipfile_serialization=False, pickle_protocol=4)
loaded_arrays = np.load('bool_arrays.npz', allow_pickle=True)
mask = []
# 从loaded_arrays中获取数组
for i in range(len(loaded_arrays)):
    array_name = 'arr_{}'.format(i)
    current_array = loaded_arrays[array_name].tolist()
    mask.append(current_array)
# print(mask)
# print(len(mask))
print(mask[0].tolist()['clip_embedding'].shape)
# print(mask[0])
# print(mask[0]['clip_embedding'])