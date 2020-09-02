# Image Classification of Flowers

- [Table of Contents](#Table_of_Contents)
  - [1. Introduction](#1-Introduction)
  - [2. File Description](#2-file-description)
  - [3. Acknowledgements](#3-Acknowledgements)

## 1. Introduction

This project is to train an image classifier with PyTorch to recognize different species of flowers. The testing accuracy of the final model is 89%.

The dataset is already split into three parts, training, validation, and testing. The project is broken down into three steps:

1. Load and preprocess the image dataset.
- For training dataset, apply transformations such as random scaling, cropping, flipping, and make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
- For validation and testing dataset, resize then crop the images to the appropriate size.
2. Train the image classifier on the training dataset
- Load a pre-trained network densenet121
- Define a new, untrained feed-forward network as a classifier, using ReLU activations
- Train the classifier layers using backpropagation using the pre-trained network to get the features
- Track the loss and accuracy on the validation set
3. Use the trained classifier to predict image content
- Save the checkpoint
- Loading the checkpoint
- Inference for classification

## 2. File Description

* `Image Classifier Project.ipynb` - the project notebook

## 3. Acknowledgements

Special thanks to Udacity for providing the dataset and for creating this project.

