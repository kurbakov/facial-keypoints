import numpy as np
import matplotlib.pyplot as plt

import torch
from models.kurbakov_net import KurbakovNet

import cv2

# load in color image for face detection
image = cv2.imread('images/obamas.jpg')

# switch red and blue color channels
# --> by default OpenCV assumes BLUE comes first, not RED as in many images
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# run the detector
# the output here is an array of detections; the corners of each detection box
# if necessary, modify these parameters until you successfully identify every face in a given image
faces = face_cascade.detectMultiScale(image, 1.2, 2)

net = KurbakovNet()
net.load_state_dict(torch.load('saved_models/keypoints_model_1_kurbakov.pt'))

image_copy = np.copy(image)

# to be sure that we have the complete face in the image
w_padding = 50
h_padding = 50

# for printing the image, we need the index
i = 0

# loop over the detected faces from your haar cascade
for (x, y, w, h) in faces:
    # Select the region of interest that is the face in the image
    roi = image_copy[y - h_padding:y + h + h_padding, x - w_padding:x + w + w_padding]

    # Convert the face region from RGB to grayscale
    image_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
    image_normalized = image_gray / 255.0

    # Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
    image_resized = cv2.resize(image_normalized, (224, 224))

    # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    image_reshaped = image_resized.reshape(image_resized.shape[0], image_resized.shape[1], 1)
    image_transposed = image_reshaped.transpose((2, 0, 1))
    torch_image = torch.from_numpy(image_transposed)

    # Make facial keypoint predictions using your loaded, trained network
    torch_image.unsqueeze_(0)
    torch_image = torch_image.type(torch.FloatTensor)

    predicted_key_pts = net(torch_image)
    predicted_key_pts = predicted_key_pts.detach().numpy()

    # we need to denormalize the data, se the class Normalize
    # Normalization has the following formula:
    # scale keypoints to be centered around 0 with a range of [-1, 1]
    # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
    predicted_key_pts = predicted_key_pts * 50.0 + 100
    predicted_key_pts = predicted_key_pts.reshape(68, 2)

    # Part to print the image
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot(1, len(faces), i + 1)
    ax.imshow(image_resized, cmap='gray')
    ax.scatter(x=predicted_key_pts[:, 0], y=predicted_key_pts[:, 1], s=30, marker='.', c='m')
    i += 1
