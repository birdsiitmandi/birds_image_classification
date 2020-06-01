import numpy as np
import pandas as pd


test = pd.read_csv("../test_image_paths.csv")
img_paths = test["image_id"]
test_labels = test.loc[:, '001.Black_footed_Albatross':'200.Common_Yellowthroat']

ts = test_labels.values

predict = np.load("predict.npy")

correct = 0
counter = 0

for i in predict[1:]:
	predicted = np.argmax(i)
	true = np.argmax(ts[counter])
	if predicted==true:
		correct +=1 
	counter+=1

print(correct)
print(correct/(counter-1))
