from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import inception_resnet_v2
from keras.applications import xception
from keras.applications.resnet import ResNet50
from keras.applications.nasnet import NASNetLarge
import numpy as np
from keras.preprocessing import image
from keras import utils
from PIL import Image
import pandas as pd
import cv2
import math
import time
import glob
from os import listdir, makedirs
from os.path import join, exists
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import efficientnet.keras as efn
from keras_efficientnets import EfficientNetB5, EfficientNetB0
import argparse
import keras.backend as K
from tta_wrapper import tta_classification

def cnn_model(model_name, img_size):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)

    if model_name == "xception":
        print("Loading Xception wts...")
        baseModel = Xception(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            input_size, weights="imagenet", include_top=False 
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            input_size, weights="imagenet", include_top=False 
        )
    elif model_name == "efn0":
        baseModel = efn.EfficientNetB0(weights="imagenet", include_top=False,
            input_shape=input_size)
    elif model_name == "efn0_noisy":
        baseModel = efn.EfficientNetB0(weights="noisy-student", include_top=False,
            input_shape=input_size)

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        200,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = False

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        # loss="categorical_crossentropy",
        loss=joint_loss,
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model


def categorical_focal_loss_fixed(y_true, y_pred, gamma, alpha):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred: A tensor resulting from a softmax
    :return: Output tensor.
    """

    # Scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Compute mean loss in mini_batch
    return K.mean(loss, axis=1)

    # return categorical_focal_loss_fixed


def cat_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def joint_loss(y_true, y_pred):
    # mse_loss = K.mean(K.square(y_true - y_pred))
    foc_loss = categorical_focal_loss_fixed(y_true, y_pred, alpha=.25, gamma=2.)
    cat_loss = K.categorical_crossentropy(y_true, y_pred)
    return foc_loss + cat_loss


# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
    # convert image into dataset
    samples = expand_dims(image, 0)
    # prepare iterator
    it = datagen.flow(samples, batch_size=n_examples)
    # make predictions for each augmented image
    yhats = model.predict_generator(it, steps=n_examples, verbose=0)
    # sum across predictions
    summed = numpy.sum(yhats, axis=0)
    # argmax across classes
    return argmax(summed)


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model_name", required=True, type=str,
        help="Imagenet model to train", default="xception"
    )
    ap.add_argument(
        "-im_size", "--image_size", type=int,
        help="Image size", default=224
    )
    ap.add_argument(
        "-w",
        "--load_weights_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    ap.add_argument(
        "-p",
        "--predictions",
        required=True,
        type=str,
        help="Predictions file name"
    )
    args = ap.parse_args()
    # Read video labels from csv file
    # files = listdir("test_frames/")
    # files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    train_data = np.load("train_data.npy")
    train_data /= 255
    train_data_mean = np.mean(train_data, axis=0)
    # print(train_data_mean.shape)
    test_dir = listdir("test/")
    test_images = []
    test_labels = []
    for sub_dir in test_dir:
        test_images += glob.glob(join("test", sub_dir, '*.jpg\n'))

    random.Random(22).shuffle(test_images)

    # print(test_images[:50])
    labels = np.load("train_label.npy")
    lb = LabelEncoder()
    onehot = lb.fit_transform(labels)

    im_size = args.image_size
    # Loading model weights
    model = cnn_model(model_name=args.model_name, img_size=im_size)
    model.load_weights(args.load_weights_name)
    print("Weights loaded...")

    if not exists("./predictions"):
        makedirs("./predictions")

    if not exists("./pred_texts"):
        makedirs("./pred_texts")

    frame_id = []
    y_predictions = []
    y_probabilities = []
    correct = 0
    total = 0
    test_label = list()

    tta_model = tta_classification(model, h_flip=True, rotation=(90, 270), 
                             h_shift=(-5, 5), merge='mean')
    for file in test_images[:]:
        # frame_id+=[file]
        img = image.load_img(file, target_size=(im_size, im_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        x -= train_data_mean
        # print(x.shape)
        # df
        # x = xception.preprocess_input(x)
        true_label = file.split('/')[-2]
        test_label += [true_label]
        # print(true_label)

        predictions = model.predict(x)
        # predictions = tta_model.predict(x)
        # print(predictions)

        predict = lb.inverse_transform([np.argmax(predictions)])[0]
        y_predictions.append(lb.inverse_transform([np.argmax(predictions)])[0])
        y_probabilities += [predictions]
        if true_label == predict:
            correct+=1

        total +=1
        if total%500==0:
            print("Number of images done:", total)
        # break
    print(correct/total)
    # print(y_predictions)


    with open( "pred_texts/" + args.predictions + '.txt', 'w') as f:
        for item in y_predictions:
            f.write("%s\n" % item)
    y_predictions = np.array(y_predictions)
    y_probabilities = np.array(y_probabilities)
    np.save("predictions/" + args.predictions + ".npy", y_predictions)

    # print(y_probabilities)
    # print(y_probabilities[:, 1])
    test_label = lb.transform(test_label)
    test_label = np.array(test_label)
    test_label = utils.to_categorical(test_label)

    y_probabilities = np.reshape(y_probabilities, (y_probabilities.shape[0], y_probabilities.shape[2]))
    # print(test_label.shape, y_probabilities.shape)
    # print(y_probabilities)
    # print(y_probabilities[:, 1])
    score = roc_auc_score(test_label, y_probabilities)
    print(score)
    # fpr, tpr, _ = roc_curve(test_label, y_probabilities)
    # roc_auc = auc(fpr, tpr)
    # print("AUC Score:", roc_auc)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.grid()
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.title('ROC Curve of kNN')
    # plt.savefig("AUC-ROC Score")


    # submission = pd.DataFrame({
    #     "Frame_ID": frame_id,
    #     "Emotion": y_predictions
    #     })
    # submission.to_csv("Test.csv", index=False)



if __name__ == "__main__":
    main()