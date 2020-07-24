import os

import numpy as np
import torch
from torch.utils.data import TensorDataset

import cv2 as cv

def load_pics(path):
    pics = os.listdir(path)
    data = [cv.imread(os.path.join(path, name)) for name in pics]
    data = np.array(data, dtype=np.float32)
    return data

def get_bg(imgs):
    return torch.from_numpy(
        np.array([cv.resize(i, (8, 8)).mean(axis=(0, 1)) for i in imgs])
    ) 

def load_SDNET2018(needBg=False):
    cache_path = './data/SDNET2018.pt'
    if os.path.exists(cache_path): 
        data = torch.load(cache_path) / 255
        if needBg: return torch.FloatTensor(data), get_bg(data)
        else: return torch.FloatTensor(data)

    paths = ['./data/SDNET2018/PD']
    data = np.vstack([load_pics(i) for i in paths])
    torch.save(data, cache_path)
    data = data / 255
    if needBg: return torch.FloatTensor(data), get_bg(data)
    else: return torch.FloatTensor(data)

if __name__ == "__main__":
    data = load_SDNET2018()
    print(data.shape)
    print(data.dtype)
    print(data.min(), data.max())