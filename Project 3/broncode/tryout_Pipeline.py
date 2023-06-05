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

# Create paths
blotch_directory = "Project 3/informatie/apple_disease_classification/Train/Blotch_Apple/" 
normal_directory = "Project 3/informatie/apple_disease_classification/Train/Normal_Apple/" 
rot_directory = "Project 3/informatie/apple_disease_classification/Train/Rot_Apple/" 
scab_directory = "Project 3/informatie/apple_disease_classification/Train/Scab_Apple/"

# Save directory
good_apples = "Project 3/informatie/apple_disease_classification/Train/good_apples/"
bad_apples = "Project 3/informatie/apple_disease_classification/Train/bad_apples/"

# Create lists
blotch_data = list()
normal_data = list()
rot_data = list()
scab_data = list()



# Read images and append to lists
for filename in os.listdir(normal_directory):
    imgFile = os.path.join(normal_directory, filename)
    normal_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(blotch_directory):
    imgFile = os.path.join(blotch_directory, filename)
    blotch_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(rot_directory):
    imgFile = os.path.join(rot_directory, filename)
    rot_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))

for filename in os.listdir(scab_directory):
    imgFile = os.path.join(scab_directory, filename)
    scab_data.append(torchvision.io.read_image(imgFile, ImageReadMode.UNCHANGED))



# Resize images
resize = T.Resize((64,64))

for i in range(len(normal_data)):
    normal_data[i] = resize(normal_data[i])

# Flip image to double the amount of images.
size = len(normal_data)
flip = T.RandomVerticalFlip(p=1)

for i in range(size):
    normal_data.append(flip(normal_data[i]))



# Rotate images
rotate90 = T.RandomRotation((90,90))
rotate180 = T.RandomRotation((180,180))
rotate270 = T.RandomRotation((270,270))

size = len(normal_data)
for i in range(size):
    normal_data.append(rotate90(normal_data[i]))
    normal_data.append(rotate180(normal_data[i]))
    normal_data.append(rotate270(normal_data[i]))


# Save images
for i in range(len(normal_data)):
    torchvision.io.write_jpeg(normal_data[i],filename=f"{good_apples}good_apple{i}.jpg")



    






