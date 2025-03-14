# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:06:17 2025

@author: User
"""

from pathlib import Path
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def find_classes(dir):
    
    dir = Path(dir)
    
    class_names = sorted(entry.name for entry in os.scandir(dir))
    if not class_names:
        raise FileNotFoundError(f"[INFO] No valid class names can be found in {dir}. Please check file structure.")
    
    class_to_idx = {}
    for idx, name in enumerate(class_names):
        class_to_idx[name] = idx
    
    return class_names, class_to_idx


class custom_data(Dataset):
    
    def __init__(self, targ_dir):
        
        targ_dir = Path(targ_dir)
        
        self.path_list = list(targ_dir.glob("*/*.png"))
        self.classes, self.class_to_idx = find_classes(targ_dir)
    
    def load_image(self, index):
        img = Image.open(self.path_list[index])
        return img
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        
        img = self.load_image(index)
        
        simpleTransform = transforms.ToTensor()
        imgTensor = simpleTransform(img)
        
        imgNorm = (2*imgTensor)-1
        
        class_name = self.path_list[index].parent.stem
        class_label = self.class_to_idx[class_name]
        
        return imgNorm, class_label