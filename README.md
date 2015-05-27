# Kaggle-Otto-Group-Product-Classification-Challenge
Authors: Vinay Rajur
---------------------------
Summary:
---------------------------
This is a classifier algorithm built as an entry for the 
Otto Group Product Classification Challenge hosted on Kaggle.com:

Competition Ended May 18th, 2015
Number of Submissions: 1
Last Submission Entry: April 24, 2015

---------------------------
Operation Instructions:
---------------------------
Download the repository and open the 'Solution Files' directory.
The script 'Run.m' will execute the entire process of loading the test 
and training data, initializing and training the neural network, and 
making classification predictions from the test data.

---------------------------
Implementation Overview:
---------------------------

A 1 hidden layer neural network implemented in MATLAB. The data supplied by
the competition contains 61,878 examples with 93 features.

Manually preprocessed training examples are loaded to memory, and the neural 
network is initialized. 

Neural network is trained on the example data, and the cost function and its
gradient are calcualted using backpropagation. Training is done using an
adaptive step-size gradient descent algorithm.

Test examples are then run through the trained neural network and
classification predictions are generated.


---------------------------
List of Key Files:
---------------------------

checkGradient.m 
--- Script to compare accuracy of gradient computed through backpropagation against
    gradient computed through numerical differentiation
    
computeCost.m
--- Function to compute the cost function and its gradient using backpropagation

feedForward.m
--- Function to feedforward a set of input data through a given neural network

gradDescent.m
--- Function to minimize an input function using adaptive step-size gradient descent

gradDescentTimed.m
--- Function to minimize an input function using adaptive step-size gradient descent
    and outputs a computation time at each step

initializer.m
--- Script to load training examples and initialize and train neural network

lambdaValidationCurves.m
--- Function to compute and plot the validation curve from various values of the 
    regularization parameter lambda and return the optimal lambda value
    
learningCurves.m
--- Script to plot the learning curves after each iteration of gradient descent

NNPredict.m
--- Function to predict classification for given neural network and input data

numericGradient.m
--- Function to compute the gradient of the cost function using numerical 
    differention

Run.m
--- Script to call initializer and compute classification predictions for test 
    examples

sigmoid.m
--- Function to evaluate the sigmoid function for given input values

TestPredicter.m
--- Function to compute predictions for test examples on a given neural network

trainNeuralNetwork.m
--- Function to train the neural network using gradient descent

Test.mat
--- Preprocessed test examples

Train.mat
--- Preprocessed training examples


  
