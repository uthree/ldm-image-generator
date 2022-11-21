import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import random
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
import numpy as np
import multiprocessing
import glob

import joblib
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir_pathes=[], cache_dir="./dataset_cache/", size=8, max_len=-1):
        super().__init__()
        self.image_path_list = []
        print("Getting paths")
        for dir_path in source_dir_pathes:
            self.image_path_list += glob.glob(os.path.join(dir_path, "**/*.jpg"), recursive=True) + glob.glob(os.path.join(dir_path, "*.png"))
        self.cache_dir = cache_dir
        self.image_path_list = self.image_path_list[:max_len]
        self.size = -1
        self.max_len = max_len
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.set_size(size)
    
    def set_size(self, size):
        if self.size == size:
            return
        self.size = size
        
        print("Initializing cache...")
        # initialize directory
        shutil.rmtree(self.cache_dir)
        # resize image and save to cache directory
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        print("Resizing images... to size: {}".format(size))
        def fn(i):
            img_path = self.image_path_list[i]
            img = Image.open(img_path)
            # get height and width
            H, W = img.size
            if H > W:
                W = int(W * size / H)
                H = size
            else:
                H = int(H * size / W)
                W = size
            flag_blur = False
            if img.size[0] > H/2 or img.size[1] > W/2:
                flag_blur = True
            img = img.resize((H, W), Image.NEAREST)
            if flag_blur:
                img = img.filter(ImageFilter.GaussianBlur(1))
            # padding to square
            empty = Image.new("RGB", (size, size), (0, 0, 0))
            # paste image to empty
            empty.paste(img, ((size - H) // 2, (size - W) // 2))
            # save to cache directory
            path = os.path.join(self.cache_dir, str(i) + ".jpg")
            empty.save(path)
            del img
            del empty

        _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(fn)(i) for i in tqdm(range(len(self.image_path_list))))
        print("Resize complete!")

    def __getitem__(self, index):
        # load image
        try:
            img_path = os.path.join(self.cache_dir, str(index) + ".jpg")
            img = Image.open(img_path)
        except Exception:
            print("Skipped error")
            img_path = os.path.join(self.cache_dir,"0.jpg")
            img = Image.open(img_path)
        # to numpy
        img = np.array(img)
        # normalize
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
            
        return img

    def __len__(self):
        return os.listdir(self.cache_dir).__len__()

class LatentImageDataset(torch.utils.data.Dataset):
    def __init__(self, source_dir_pathes=[], cache_dir="./dataset_cache/", size=512, max_len=-1, encoder=torch.nn.Identity(), device=torch.device('cpu'), n_workers=1):
        super().__init__()
        self.image_path_list = []
        print("Getting paths")
        for dir_path in source_dir_pathes:
            self.image_path_list += glob.glob(os.path.join(dir_path, "**/*.jpg"), recursive=True) + glob.glob(os.path.join(dir_path, "*.png"))
        self.cache_dir = cache_dir
        self.image_path_list = self.image_path_list[:max_len]
        self.size = -1
        self.max_len = max_len
        self.encoder = encoder
        self.device = device
        self.n_workers = n_workers
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.set_size(size)

    def set_size(self, size):
        print("Setting up Encoder...")
        if self.size == size:
            return
        self.size = size
        self.encoder.to(self.device)
        
        print("Initializing cache...")
        # initialize directory
        shutil.rmtree(self.cache_dir)
        # resize image and save to cache directory
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        print("Resizing images... to size: {}".format(size))
        @torch.no_grad()
        def fn(i):
            img_path = self.image_path_list[i]
            img = Image.open(img_path)
            # get height and width
            H, W = img.size
            if H > W:
                W = int(W * size / H)
                H = size
            else:
                H = int(H * size / W)
                W = size
            flag_blur = False
            if img.size[0] > H/2 or img.size[1] > W/2:
                flag_blur = True
            img = img.resize((H, W), Image.NEAREST)
            if flag_blur:
                img = img.filter(ImageFilter.GaussianBlur(1))
            # padding to square
            empty = Image.new("RGB", (size, size), (0, 0, 0))
            # paste image to empty
            empty.paste(img, ((size - H) // 2, (size - W) // 2))
            # concatenate path
            path = os.path.join(self.cache_dir, str(i) + ".pt")
            # to numpy
            img = img.convert("RGB")
            img = np.array(img)
            # normalize
            img = img / 127.5 - 1.0
            img = np.transpose(img, (2, 0, 1))
            img = img.astype(np.float32)
            # Convert to torch.tensor
            img = torch.tensor(img).to(self.device)
            img = img.unsqueeze(0)
            # encode
            with torch.no_grad():
                z, _ = self.encoder(img)
            torch.save(z, path)
            del img
            del empty

        _ = joblib.Parallel(n_jobs=self.n_workers)(joblib.delayed(fn)(i) for i in tqdm(range(len(self.image_path_list))))
        self.encoder.to(torch.device('cpu'))
        print("Resize and encode complete!")

    def __getitem__(self, index):
        # load image
        try:
            img_path = os.path.join(self.cache_dir, str(index) + ".pt")
            img = torch.load(img_path)
        except Exception:
            print("Skipped error")
            img_path = os.path.join(self.cache_dir,"0.pt")
            img = torch.load(img_path)
            
        return img[0]

    def __len__(self):
        return os.listdir(self.cache_dir).__len__()

