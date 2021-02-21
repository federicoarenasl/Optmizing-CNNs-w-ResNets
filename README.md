# Optimizing Convolutional Neural Networks with Residual Blocks and Batch Normalization
During this study we will identify what causes the vanishing/exploding gradient problem present in deep Convolutional Neural Networks by comparing the learning performance of two VGG architectures trained on the CIFAR-100 datset, one shallow (8 convolutional layers), and one deep (38 convolutional layer). We then move on to explore 3 possible solutions present in the literature. From these solutions we choose to implement Deep Residual Networks on the VGG architecture, and test the solution under multiple hyperparameter scenarios. We finally train the best VGG ResNet architecture on increasing depths, and find that a 68 layered network provides the best performance yielding a test accuracy of 65.83 %.

## The CIFAR100 Dataset
All of these experiments will be done on the benchmark balanced EMNIST dataset [(Cohen et al., 2017)](https://arxiv.org/pdf/1702.05373.pdf). This dataset contains 131,600 28x28 px images of 47 different handwritten digits. The training set counts with 100,000 images, the validation set counts 15,800 images, and the test set counts with 15,800 images. At each experiment, the networks will be fed the training set, validated and fine-tuned in the validation set, and the best network will be evaluated on the test set. More on this [here](https://www.nist.gov/itl/products-and-services/emnist-dataset).

## The MLP machine learning framework
For constructing our Neural Network architectures we use the MLP ML Framework, since this study was done within the context of the Machine Learning Practical Course at the University of Edinburgh. The framework is not made available due to school licensing purposes.

## Structure of the post
  - Evaluation_of_Regularisation_Techniques.pdf : This file represents the main file of the post, and we strongly encourage the reader to start by giving it a quick read to understand the project better. If the reader is still curios as to _how_ the results in the study were obtained, we encourage the reader to checkout the next two files.
  - Training-w-mlp-framework.md : this file, accompanied by its ```.ipynb``` version in the ```notebooks``` folder, will walk the reader through the bulding of the Neural Network architectures and the training of these networks with Dropout, L1 Regularisation, and L2 Regularisation. It will also include the hyperparameter search for the best model, and the training and testing of it.
  - Training-Results-Visualizer.mp : this file, accompanied by its ```.ipynb``` version in the ```notebooks``` folder, will go through the plotting and brief analysis of the training results, as well as reporting the test results from the last best model.
 
 ## A sneak peak at some results
Our first network, purposedly designed to show bad generalization performance is a Network for 100 Epochs, using Stocahstic Gradient Descent and Adam optimizer with a mini-batch size of 100, with one Affine Layer composed of 100 Hidden Units followed by a ReLu non-linearity, with learnng rate of 0.001 and all biases and weights initialised to 0. The generalization problem is evident as illustrated by the following figure.

<p align="center">
<img  src="Training-Results-Visualizer_files/Training-Results-Visualizer_8_0.png">
</p>

After a thorough hyperparameter search, we are able to find a model that, solely with regularisation, (1) greatly lower the Train/Test Error Gap from a 1.42 to a 0.13 Gap. Lower the Train/Test Accuracy Gap which went from a 14% to a 3.57% Accuracy Gap. Additionally, (2) we were able to increase the Test Accuracy from 81.4% to 84.03%. The Final Model is able to stably converge to a local minimum after 15 Epochs of training:

<p align="center">
<img  src="Training-Results-Visualizer_files/Training-Results-Visualizer_74_0.png">
</p>



