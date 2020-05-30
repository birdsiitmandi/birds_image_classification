from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet import ResNet50
from keras.applications.nasnet import NASNetLarge
from matplotlib import pyplot as plt
from keras import backend as K
from keras import utils
import numpy as np
import time
import argparse
from os.path import exists
from os import makedirs
import efficientnet.keras as efn
# from clr_callback import CyclicLR
from keras_efficientnets import EfficientNetB5, EfficientNetB0
# from random_eraser import get_random_eraser  # added


# MIN_LR = 1e-7
# MAX_LR = 1e-2
# STEP_SIZE = 8
# CLR_METHOD = "triangular"


def cnn_model(model_name, img_size, nb_classes, model_no=7):
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
    # elif model_name == "ef0":
    #     baseModel = EfficientNetB0(
    #         input_size, weights="imagenet", include_top=False 
    #     )
    # elif model_name == "ef5":
    #     baseModel = EfficientNetB5(
    #         input_size, weights="imagenet", include_top=False 
    #     )
    elif model_name == "efn":
        baseModel = efn.EfficientNetB7(weights="imagenet", include_top=False,
            input_shape=input_size)
    elif model_name == "efn_noisy":
        baseModel = efn.EfficientNetB7(weights="noisy-student", include_top=False,
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
        nb_classes,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        # loss="categorical_crossentropy",
        loss=joint_loss,
        optimizer='adam',
        metrics=["accuracy"]
    )
    return model


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

    
# def categorical_focal_loss(gamma=2., alpha=.25):
#     """
#     Softmax version of focal loss.
#            m
#       FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
#           c=1
#       where m = number of classes, c = class and o = observation
#     Parameters:
#       alpha -- the same as weighing factor in balanced cross entropy
#       gamma -- focusing parameter for modulating factor (1-p)
#     Default value:
#       gamma -- 2.0 as mentioned in the paper
#       alpha -- 0.25 as mentioned in the paper
#     References:
#         Official paper: https://arxiv.org/pdf/1708.02002.pdf
#         https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
#     Usage:
#      model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
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


def build_lrfn(lr_start=0.00001, lr_max=0.000075, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-e", "--epochs", type=int,
        help="Number of epochs", default=25
    )
    ap.add_argument(
        "-m", "--model_name", required=True, type=str,
        help="Imagenet model to train", default="xception"
    )
    ap.add_argument(
        "-b", "--batch_size", required=True, type=int,
        help="Batch size", default=8
    )
    ap.add_argument(
        "-im_size", "--image_size", type=int,
        help="Batch size", default=224
    )
    ap.add_argument(
        "-n_class", "--n_classes", type=int,
        help="Number of classes", default=200
    )
    ap.add_argument(
        "-w",
        "--weights_save_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    args = ap.parse_args()

    # Training dataset loading
    train_data = np.load("../train_data.npy")
    train_label = np.load("../train_label.npy")
    # print(train_label[:60])
    encoder = LabelEncoder()
    encoder.fit(train_label)
    encoded_y = encoder.transform(train_label)
    # print(encoded_y[:60])
    # ls = set(encoded_y)
    # print(ls)
    Y = utils.to_categorical(encoded_y)

    print("Dataset Loaded...")

    # Train and validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, Y, test_size=0.2, shuffle=True, random_state=42, stratify=Y
    )
    print(trainX.shape, valX.shape, trainY.shape, valY.shape)

    trainX /= 255
    valX /= 255
    trainX_mean = np.mean(trainX, axis=0)
    trainX -= trainX_mean
    valX -= trainX_mean

    # Train nad validation image data generator
    # trainAug = ImageDataGenerator(
    #     rescale=1.0 / 255.0,
    #     # preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
    #     #           v_l=0, v_h=255, pixel_level=False),
    #     rotation_range=30,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest",
    # )

    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=30,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.15,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    model = cnn_model(args.model_name, img_size=args.image_size, nb_classes=args.n_classes)

    # Number of trainable and non-trainable parameters
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    )
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    if not exists("./trained_wts"):
        makedirs("./trained_wts")
    # if not exists("./training_logs"):
    #     makedirs("./training_logs")
    if not exists("./plots"):
        makedirs("./plots")

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/" + args.weights_save_name + ".hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=3,
                                   min_lr=0.5e-6)
    lrfn = build_lrfn()
    lr_schedule = LearningRateScheduler(lrfn, verbose=1)

    # clr = CyclicLR(
    #     mode = CLR_METHOD,
    #     base_lr = MIN_LR,
    #     max_lr = MAX_LR,
    #     step_size = STEP_SIZE * (trainX.shape[0] // args.batch_size)
    # )
    print("Training is going to start in 3... 2... 1... ")

    datagen.fit(trainX)
    # Model Training
    H = model.fit_generator(
        datagen.flow(trainX, trainY, batch_size=args.batch_size),
        steps_per_epoch=len(trainX) // args.batch_size,
        validation_data=(valX, valY),
        validation_steps=len(valX) // args.batch_size,
        epochs=args.epochs,
        workers=4,
        callbacks=[model_checkpoint, lr_reducer],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = args.epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

    # N = np.arange(0, len(clr.history["lr"]))
    # plt.figure()
    # plt.plot(N, clr.history["lr"])
    # plt.title("Cyclical Learning Rate (CLR)")
    # plt.xlabel("Training Iterations")
    # plt.ylabel("Learning Rate")
    # plt.savefig("plots/cyclic_lr.png")

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()