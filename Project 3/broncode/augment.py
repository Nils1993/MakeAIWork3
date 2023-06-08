# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision.io import ImageReadMode
from torchvision.io import read_image
import matplotlib.pyplot as plt
import os
import math

# Save directory
# good_apples = "Project 3/informatie/apple_disease_classification/images/Train/good_apples/"
blotch_apples = "Project 3/informatie/apple_disease_classification/images/Train/dataset/blotch_apples/"
rot_apples = "Project 3/informatie/apple_disease_classification/images/Train/dataset/rot_apples/"
scab_apples = "Project 3/informatie/apple_disease_classification/images/Train/dataset/scab_apples/"

# Create lists
# good_data = list()
blotch_data = list()
rot_data = list()
scab_data = list()

# good_filenames = list()
blotch_filenames = list()
rot_filenames = list()
scab_filenames = list()

# Read images and append to lists
# for filename in os.listdir(good_apples):
#     imgFile = os.path.join(good_apples, filename)
#     good_filenames.append(filename)
#     good_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(blotch_apples):
    imgFile = os.path.join(blotch_apples, filename)
    blotch_filenames.append(filename)
    blotch_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(rot_apples):
    imgFile = os.path.join(rot_apples, filename)
    rot_filenames.append(filename)
    rot_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(scab_apples):
    imgFile = os.path.join(scab_apples, filename)
    scab_filenames.append(filename)
    scab_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))


# Flip image to double the amount of images.
flip = T.RandomVerticalFlip(p=1)

# for i in range(len(good_data)):
#     good_data.append(flip(good_data[i]))
for i in range(len(blotch_data)):
    blotch_data.append(flip(blotch_data[i]))

for i in range(len(rot_data)):
    rot_data.append(flip(rot_data[i]))

for i in range(len(scab_data)):
    scab_data.append(flip(scab_data[i]))




# Rotate images
rotate90 = T.RandomRotation((90,90))
rotate180 = T.RandomRotation((180,180))
rotate270 = T.RandomRotation((270,270))

# for i in range(len(good_data)):
#     good_data.append(rotate90(good_data[i]))
#     good_data.append(rotate180(good_data[i]))
#     good_data.append(rotate270(good_data[i]))

for i in range(len(blotch_data)):
    blotch_data.append(rotate90(blotch_data[i]))
    blotch_data.append(rotate180(blotch_data[i]))
    blotch_data.append(rotate270(blotch_data[i]))

for i in range(len(rot_data)):
    rot_data.append(rotate90(rot_data[i]))
    rot_data.append(rotate180(rot_data[i]))
    rot_data.append(rotate270(rot_data[i]))

for i in range(len(scab_data)):
    scab_data.append(rotate90(scab_data[i]))
    scab_data.append(rotate180(scab_data[i]))
    scab_data.append(rotate270(scab_data[i]))



# Save images
# for i in range(len(good_data)):
#     torchvision.io.write_jpeg(good_data[i],filename=f"{good_apples}good_apple{i}.jpg")

for i in range(len(blotch_data)):
    torchvision.io.write_jpeg(blotch_data[i],filename=f"{blotch_apples}blotch_apple_aug{i}.jpg")

for i in range(len(rot_data)):
    torchvision.io.write_jpeg(rot_data[i],filename=f"{rot_apples}rot_apple_aug{i}.jpg")

for i in range(len(scab_data)):
    torchvision.io.write_jpeg(scab_data[i],filename=f"{scab_apples}scab_apple_aug{i}.jpg")




    






