# Multi-class-classifiers
Here we compare different classifiers and their performance for multi-class classications

We use MNIST dataset to compare different classifiers and their performance. The MNIST database of handwritten digits has a training set of 60,000 examples and a test set of 10,000 examples.
We compare KNN, Logistic Regression, SVM, kernel SVM, and Neural Neuwroks.

First we standardize the features before training the classifiers by dividing the values of the features by 255. 

We use cross-validation (CV) to find the optimal number of neighbors in KNN.

For Neural Netweorks (NN), we use 10 hidden layers.

For kernel SVM, we use radial basis function and choose the proper kernel.

For KNN and SVM we randomly downsample the training data to size m=5000 to improve the computation efficiency.


