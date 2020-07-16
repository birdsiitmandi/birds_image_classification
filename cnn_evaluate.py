from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
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
# from sklearn.metrics import confusion_matrix
import random
import efficientnet.keras as efn
import argparse
import keras.backend as K
from tta_wrapper import tta_classification
from imgaug import augmenters as iaa

def cnn_model(model_name, img_size):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)

    if model_name == "efn0":
        baseModel = efn.EfficientNetB0(weights="imagenet", include_top=False,
            input_shape=input_size)
    elif model_name == "efn_noisy":
        baseModel = efn.EfficientNetB5(weights="noisy-student", include_top=False,
            input_shape=input_size)

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(1024, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
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
        "-m", "--model_name", type=str,
        help="Imagenet model to train", default="efn_noisy"
    )
    ap.add_argument(
        "-im_size", "--image_size", type=int,
        help="Image size", default=299
    )
    ap.add_argument(
        "-w",
        "--load_weights_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    # ap.add_argument(
    #     "-p",
    #     "--predictions",
    #     required=True,
    #     type=str,
    #     help="Predictions file name"
    # )
    args = ap.parse_args()
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip
        sometimes(iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)))
    ], random_order=True)

    train_data_mean = np.load("train_data_mean_299.npy")

    test_dir = listdir("../test/")
    test_images = []
    test_labels = []
    for sub_dir in test_dir:
        test_images += glob.glob(join("../test", sub_dir, '*.jpg\n'))

    random.Random(22).shuffle(test_images)

    labels = np.load("../train_label.npy")
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
    
    y_probabilities = []
    correct = 0
    total = 0
    test_label = list()

    tta_model = tta_classification(model, h_flip=True, rotation=(90, 270), 
                             h_shift=(-5, 5), merge='mean')

    y_predictions = []
    for file in test_images[:]:
        # frame_id+=[file]
        img = image.load_img(file, target_size=(im_size, im_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        x -= train_data_mean

        true_label = file.split('/')[-2]
        test_label += [true_label]

        predictions = model.predict(x)
        # predictions = tta_model.predict(x)

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


if __name__ == "__main__":
    main()
