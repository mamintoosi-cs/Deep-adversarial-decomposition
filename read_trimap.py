#  M. Amintoosi
#  Read trimap image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import argparse

import utils

# import torch
import torchvision.transforms.functional as TF

datadir = r'./images/custom'
val_dirs = glob.glob(os.path.join(datadir, 'dog*.jpg'))
datadir_flowers = r'./images/custom'
val_dirs_flowers = glob.glob(os.path.join(datadir_flowers, 'flower*.jpg'))

img1 = cv2.imread(val_dirs[idx], cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread(val_dirs_flowers[idx], cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


# we recommend to use TF.resize since it was also used during trainig
# You may also try cv2.resize, but it will produce slightly different results
gt1 = TF.resize(TF.to_pil_image(img1), [args.in_size, args.in_size])
gt1 = 0.5*TF.to_tensor(gt1).unsqueeze(0)
gt2 = TF.resize(TF.to_pil_image(img2), [args.in_size, args.in_size])
gt2 = 0.5*TF.to_tensor(gt2).unsqueeze(0)
img_mix = gt1 + gt2

cv2.imshow(img_mix)
