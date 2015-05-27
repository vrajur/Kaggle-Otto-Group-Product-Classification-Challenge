% learningCurves.m
% Plot Learning Curves after each iteration of gradient descent


function trained_thetas = learningCurves(thetas, dims, X, y, valSize, lambda)

m = size(X,1);
indsTrain = randperm(m, ciel(m/2));
Xtrain = X(indsTrain,:);
ytrain = y(indsTrain,:);

indsVal = setdiff(1:m,indsTrain);
XVal = X(indsVal,:);
yval = y(indsVal,:);

handle = figure(1); hold on;
title(sprintf('Learning Curves for lambda = %d, m_train = %d', lambda, length(indsTrain)));
xlabel('Training Examples');
ylabel('Error');

trained_thetas = gradDescentLC(@(t)computeCost(t, dims, X, y, lambda), thetas, 400, handle);




end