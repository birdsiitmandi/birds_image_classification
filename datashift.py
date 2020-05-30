import os
from os.path import exists, join, split
from shutil import copyfile

split_file = open("train_test_split.txt", "r")
# print(split_file.readline())

train = "./train/"
test = "./test/"

classes = open("classes.txt", "r")
images = open("images.txt", "r")
labels = open("image_class_labels.txt", "r")

train_counter = 0
test_counter = 0
for (x, y) in zip(split_file, images):
	# print(x, y)
	image_id = x.split(" ")[0]
	is_train = x.split(" ")[1]
	# print(image_id, is_train)
	class_id = y.split(" ")[1].split("/")[0]
	image_name = y.split(" ")[1].split("/")[1]
	# print(class_id, image_name)

	if not exists(train + class_id):
		os.makedirs(train + class_id)

	if not exists(test + class_id):
		os.makedirs(test + class_id)
	
	# print(join("images", y.split(" ")[1]))
	# print(join(train, class_id, image_name))
	if int(is_train)==1:
		copyfile(join("images", y.split(" ")[1][:-1]), join(train, class_id, image_name))
		train_counter+=1
	else:
		copyfile(join("images", y.split(" ")[1])[:-1], join(test, class_id, image_name))
		test_counter+=1
	# if train_counter==5:
	# 	break
	# break

print(train_counter, test_counter)