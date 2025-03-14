# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:57:13 2025

@author: User
"""

from pathlib import Path
import _csv as csv
import numpy as np
import os
import cv2

def extract_mnist(save_dir, csv_fname):
    
    assert os.path.exists(csv_fname), f"[INFO] {csv_fname} cannot be found. Please check the file structure."
    
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    with open(csv_fname) as f:
        reader = csv.reader(f)
        
        for idx, row in enumerate(reader):
            
            if idx == 0:
                continue
            
            img = np.zeros((28*28))
            img[:] = list(map(int, row[1:]))
            img = img.reshape((28,28))
            
            save_class_folder = save_dir / row[0]
            if not save_class_folder.is_dir():
                save_class_folder.mkdir(parents = True,
                                        exist_ok = True)
            
            save_image_path = save_class_folder / f"{idx}.png"
            cv2.imwrite(save_image_path, img)
            
            if (idx+1) % 1000 == 0:
                print(f"[INFO] {idx+1} images have been saved into save_dir.")

extract_mnist(save_dir = "./data/train",
              csv_fname = "./rawData/mnist_train.csv")

extract_mnist(save_dir = "./data/test",
              csv_fname = "./rawData/mnist_test.csv")


            
    