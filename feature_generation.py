from keras.models import Model
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
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
import random
import efficientnet.keras as efn
import argparse
import keras.backend as K


def cnn_model(model_name, img_size, weights):
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
    headModel = Dense(1024, activation="relu", kernel_initializer="he_uniform", name="fc1")(
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
    model.load_weights(weights)

    model_fc = Model(
        inputs=baseModel.input,
        outputs=model.get_layer("fc1").output
    )

    for layer in baseModel.layers:
        layer.trainable = False

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        # loss=joint_loss,
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model_fc


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


def cat_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def joint_loss(y_true, y_pred):
    # mse_loss = K.mean(K.square(y_true - y_pred))
    foc_loss = categorical_focal_loss_fixed(y_true, y_pred, alpha=.25, gamma=2.)
    cat_loss = K.categorical_crossentropy(y_true, y_pred)
    return foc_loss + cat_loss


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model_name", type=str,
        help="Imagenet model to train", default="efn_noisy"
    )
    ap.add_argument(
        "-im_size", "--image_size", type=int,
        help="Image size", default=224
    )
    ap.add_argument(
        "-t", "--train", type=int,
        help="Image size", default=1
    )
    ap.add_argument(
        "-w",
        "--load_weights_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    args = ap.parse_args()

    train_data_mean = np.load("train_data_mean_299.npy")
    images = []

    if args.train == 1:
        print("Training feature generation")
        train_dir = listdir("../train/")
        
        for sub_dir in train_dir:
            images += glob.glob(join("../train", sub_dir, '*.jpg\n'))


    else:
        print("Testing feature generation ...")
        test_dir = listdir("../test/")
        
        for sub_dir in test_dir:
            images += glob.glob(join("../test", sub_dir, '*.jpg\n'))


    random.Random(22).shuffle(images)

    # print(test_images[:50])
    labels = np.load("../train_label.npy")
    lb = LabelEncoder()
    onehot = lb.fit_transform(labels)

    im_size = args.image_size
    # Loading model weights
    model = cnn_model(model_name=args.model_name, img_size=im_size, weights=args.load_weights_name)
    print("Model loaded...")

    features = []
    train_labels = []
    counter = 0

    for file in images[:]:
        img = image.load_img(file, target_size=(im_size, im_size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        x -= train_data_mean

        true_label = file.split('/')[-2]
        true_label = lb.transform([true_label])
        train_labels += [true_label[0]]

        predictions = model.predict(x)
        features += [predictions]

        if counter%500==0:
            print("Number of images done:", counter)
        counter += 1 

    features = np.array(features)
    features = np.reshape(features, (features.shape[0], features.shape[2]))
    train_labels = np.array(train_labels)

    # print(features.shape, train_labels.shape)
    # print(train_labels)
    if args.train == 1:
        np.save("train_vec.npy", features)
        np.save("train_tr_labels.npy", train_labels)
    else:
        np.save("test_vec.npy", features)
        np.save("test_tr_labels.npy", train_labels)


    # with open( "pred_texts/" + args.predictions + '.txt', 'w') as f:
    #     for item in y_predictions:
    #         f.write("%s\n" % item)
    # y_predictions = np.array(y_predictions)
    # y_probabilities = np.array(y_probabilities)
    # np.save("predictions/" + args.predictions + ".npy", y_predictions)



if __name__ == "__main__":
    main()