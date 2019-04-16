import numpy as np
import matplotlib.pyplot as plt

import torch
import cv2

from models.kurbakov_net import KurbakovNet


net = KurbakovNet()
net.load_state_dict(torch.load('saved_models/keypoints_model_1_kurbakov.pt'))

# Step 1: Check how the kernel looks like
# Get the weights in the first conv layer, "conv1"
# and visualize the kernel
weights = net.conv1.weight.data
w = weights.numpy()

print(w[0][0])
print(w[0][0].shape)
plt.imshow(w[0][0], cmap='gray')

# Step 2: Check how the filters effect the image
# load in and display any image from the transformed test dataset
original_image = cv2.imread('images/obamas.jpg')
image = original_image.numpy()
image = np.transpose(image, (1, 2, 0))


filtered_image = cv2.filter2D(image, -1, w[0][0])

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.set_title('original image')
ax1.imshow(original_image[0], cmap='gray')

ax2.set_title('filter 0 - image in conv2')
ax2.imshow( cv2.filter2D(image, -1, w[0][0]), cmap='gray')