% trainNeuralNetwork.m
% Train the Neural Network by Minimizing its Cost Function

function trained_thetas = trainNeuralNetwork(thetas, dims, X, y, lambda)


trained_thetas = gradDescent(@(t)computeCost(t, dims, X, y, lambda), thetas, 400);

% opt = ['GradObj', 'on', 'MaxIter', 400];
% trained_thetas = fmincg(@(t)computeCost(t, dims, X, y, lambda), thetas, opt);



end