from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import VotingClassifier
import pickle

is_train = 1

trainX = np.load("train_vec.npy")
Y_train = np.load("train_tr_labels.npy")

testX = np.load("test_vec.npy")
Y_test = np.load("test_tr_labels.npy")

sc = StandardScaler()

X_train = sc.fit_transform(trainX)
X_test = sc.transform(testX)

model_1 = LogisticRegression(random_state=1)
model_2 = RandomForestClassifier(random_state=1, n_estimators=100)
# model_3 = SVC(probability=True, random_state=1)
# model_3 = LinearSVC(random_state=1)
model_4 = KNeighborsClassifier(n_neighbors=3)

model = VotingClassifier(estimators=[('lr', model_1), ('ls', model_2), ('knc', model_4)], voting='soft')


if is_train==1:

	model.fit(X_train, Y_train)
	with open('voting_classifier.pkl', 'wb') as files:
		pickle.dump(model, files)

else:

	with open('voting_classifier.pkl', 'rb') as files:
		voting_loaded = pickle.load(files)

	y_pred = voting_loaded.predict(X_test)
	print(accuracy_score(Y_test, y_pred))