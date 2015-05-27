% Run.m
% Script to Run Initialization of Neural Network, Training of Neural
% Network and Predictions of Neural Network on Test Set

initializer;

trained_thetas = trainNeuralNetwork(thetas,dims, X, y, 0);

TestPredicter(trained_thetas,dims);