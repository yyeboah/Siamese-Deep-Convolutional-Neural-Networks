# Siamese Convolutional Neural Networks
 A Siamese Convolutional Neural Network implementation in Keras (TensorFlow Backend)

## Design Overview
 The network is designed with a VGG16 backbone (pretrained on Imagenet) with two output branches. In the first branch the network performs per-pixel semantic labeling (semantic segmentation) while in the second network branch, image classification class labels are output. The network accepts a single image as input at inference time.

## Implementation Environment
This code is implemented in Pythpn 2.7 with tensorflow 1.3.0. It has been tested on Ubuntu 16.04 with CUDA 8 / CuDNN v5. 