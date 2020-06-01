# birds_image_classification
Fine-grained species classification

In this work, we are trying to improve the fine-grained classification on Caltech UCSD Birds dataset. We are publishing codes in both Keras and Pytorch framework, so that it's easy to reproduce by anyone.

## Table of Contents

- [Requirements](#requirements)
- [Caltech-UCSD-2011](#cub)
- [Results](#results)
- [Citation](#citation)
- [Notes](#notes)

## Requirements

Tested on Python 3.6.x and Keras 2.3.0 with TF backend version 1.14.0.
* Numpy (1.16.4)
* OpenCV (4.1.0)
* Pandas (0.25.3)
* Scikit-learn (0.22.1)
* PyTorch (1.2.0)

## Getting Started

### Keras
* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
 ```
* [cnn_train.py](https://github.com/birdsiitmandi/birds_image_classification/blob/master/keras/cnn_train.py) - Transfer learning with EF Net
* [cnn_evaluate.py](https://github.com/birdsiitmandi/birds_image_classification/blob/master/keras/cnn_evaluate.py) - Evaluation on birds dataset

### Pytorch
* [cnn_train.py](https://github.com/birdsiitmandi/birds_image_classification/blob/master/pytorch/cnn_train.py) - Transfer learning with EF Net
* [model_evaluate.py](https://github.com/birdsiitmandi/birds_image_classification/blob/master/pytorch/model_evaluate.py) - Evaluation on birds dataset
