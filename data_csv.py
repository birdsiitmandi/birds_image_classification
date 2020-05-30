import cv2
import csv
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import glob
import os
from os.path import isfile, join, split
from os import rename, listdir, rename, makedirs
from random import shuffle
import random
import glob
import pickle
import matplotlib.pyplot as plt

train_dir = listdir("test")
# print(train_dir)
images = []

for sub_dir in train_dir:
	images += glob.glob(join("test", sub_dir, '*.jpg\n'))

random.Random(64).shuffle(images)

# print(images[:50])

with open("test_image_paths.csv", "w") as f:
	writer = csv.writer(f)
	writer.writerow(["image_id", "label"])
	for i in images[:]:
		img_path = i
		# print(img_path)
		label = i.split("/")[1]
		# print(label)
		# brd
		writer.writerow([img_path, label])

