# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:14:56 2025

@author: User
"""

import torch
import torch.nn as nn
from load_data import custom_data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import VAE
# from torchinfo import summary
from engine import train_step, test_step
from tqdm.auto import tqdm
from pathlib import Path
from einops import rearrange
import numpy as np
import torchvision

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path
trainPath = "./data/train"
testPath = "./data/test"
resultPath = "./result"

# Hyperparmeter
EPOCHS = 10
LEARNING_RATE = 1E-3
BATCH_SIZE = 32

# 1. Create custom dataset
trainData = custom_data(targ_dir = trainPath)
testData = custom_data(targ_dir = testPath)

# 2. Visualize the data

randNum = torch.randint(0, len(trainData)-1, (9,))

for idx, num in enumerate(randNum):
    
    trainImg, trainLabel = trainData[num]
    
    trainImgPlt = ((trainImg.squeeze())+1)/2
    
    plt.subplot(3,3,idx+1)
    plt.imshow(trainImgPlt, cmap="gray")
    plt.title(f"Label: {trainLabel}")
    plt.axis(False)

plt.tight_layout()
plt.show()

print(f"[INFO] The size of an image                     : {trainImg.shape}")
print(f"[INFO] The range of the values within the image : {trainImg.min()} to {trainImg.max()}")
print(f"[INFO] The number of images within the dataset  : {len(trainData)}")
print(f"[INFO] Classes available                        : {trainData.classes}")


# 3. Create dataloader

trainDataLoader = DataLoader(dataset = trainData,
                             batch_size = BATCH_SIZE,
                             shuffle = True)

testDataLoader = DataLoader(dataset = testData,
                            batch_size = BATCH_SIZE,
                            shuffle = False)

# 4. Visualize the dataloader
trainImgBatch, trainLabelBatch = next(iter(trainDataLoader))

print(f"[INFO] Total number of batches        : {len(trainDataLoader)}")
print(f"[INFO] Number of images in one batch : {len(trainImgBatch)}")
print(f"[INFO] Size of an image              : {trainImgBatch[0].shape}")

# 5. Instantiate a model
model0 = VAE().to(device)

# 6. Verify the model
# =============================================================================
# summary(model = model0,
#         input_size = (1,1,28,28),
#         col_names = ["input_size", "output_size", "num_params", "trainable"],
#         row_settings = ["var_names"])
# =============================================================================

# 7. Create optimizer
optimizer = torch.optim.Adam(params = model0.parameters(),
                             lr = LEARNING_RATE)

# 8. Create training loop
trainLossList = []
testLossList = []
    
for epoch in tqdm(range(EPOCHS)):
    
    trainResult = train_step(model = model0,
                             dataloader = trainDataLoader,
                             device = device,
                             optimizer = optimizer)
    
    testResult = test_step(model = model0,
                           dataloader = testDataLoader,
                           device = device)
    
    trainLossList.append(trainResult['loss'].cpu().detach().numpy())
    testLossList.append(testResult['loss'].cpu().detach().numpy())
    
    print(f"[INFO] Current epoch: {epoch} ")
    print(f"[INFO] Train Loss : {trainResult['loss']:.4f} | Test Loss: {testResult['loss']:.4f}")

# 9. Plot the result
plt.plot(range(EPOCHS), trainLossList, color = "red", label = "trainLoss")
plt.plot(range(EPOCHS), testLossList, color = "blue", label = "testLoss")
plt.title("Loss")
plt.xticks(np.arange(0, EPOCHS, 1))
plt.legend()
plt.show()


####################### Separate here in Colab ############################

# 10. Save the model
def save_model(model, save_dir, save_name):
    
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    assert save_name.endswith(".pt") or save_name.endswith(".pth"), f"[INFO] {save_name} is not valid. Please check the file format / extension."
    save_file = save_dir / save_name
    
    torch.save(obj = model.state_dict(),
               f = save_file)

save_model(model = model0,
           save_dir = "./model",
           save_name = "model0.pt")


# 11. Load the model
model1 = VAE().to(device)

model1.load_state_dict(torch.load(f = "./model/model0.pt",
                                  weights_only = True ))

# 12. Make inference

idxs = torch.randint(0, len(testData)-1, (100,))
testImgs = torch.cat([testData[idx][0][None, :].to(device) for idx in idxs])

model1.eval()
with torch.inference_mode():
    generatedImgs, _, _ = model1(testImgs)

testImgs = (testImgs+1)/2
generatedImgs = 1 - (generatedImgs+1)/2
 
out = torch.hstack([testImgs, generatedImgs])
output = rearrange(out, "B C H W -> B () H (C W)")

grid = torchvision.utils.make_grid(output, nrow=10)
result = torchvision.transforms.ToPILImage()(grid)

resultPath = Path(resultPath)
if not resultPath.is_dir():
    resultPath.mkdir(parents = True,
                 exist_ok = True)

resultFile = resultPath / "reconstruction.jpg"
result.save(resultFile)

    
    
    









