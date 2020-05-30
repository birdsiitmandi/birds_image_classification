import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import glob
import os
from os.path import isfile, join, split
from os import rename, listdir, rename, makedirs
from random import shuffle
import glob
import pickle
import matplotlib.pyplot as plt

train_dir = listdir("train")
# print(train_dir)
images = []

for sub_dir in train_dir:
	# for img in listdir(join("train", sub_dir)):
		# im = pickle.load(img)
		# print(img)
		# im = cv2.imread(join("train", sub_dir, img))
		# # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
		# print(im.shape)
		# plt.imshow(im)
		# plt.savefig("fig.png")
		# break
	# print(listdir(join("train", sub_dir)))
	# print(glob.glob(join("train", sub_dir, '*.jpg\n')))
	images += glob.glob(join("train", sub_dir, '*.jpg\n'))

# print(images)
# print(len(images))

train_data = []
train_label = []
count = 0
for j in images:

	img = cv2.imread(j)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224, 224))
	train_data.append(img)
	train_label+=[j.split('/')[-2]]

	if count%1000==0:
		print("Number of images done:", count)
	count+=1

# print(train_label)

train_data = np.array(train_data).astype("float32")
train_label = np.array(train_label)
# print(train_data.dtype)
# print(train_data[0])
# print(train_data[0]/255)
# l = train_data[0]/255
# print(l.dtype)
# g = train_data[0].astype("float")
# print(g/255)

print(train_data.shape, train_label.shape)
np.save("train_data.npy", train_data)
np.save("train_label.npy", train_label)