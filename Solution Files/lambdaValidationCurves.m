% lambdaValidationCurves.m
% Plot the Validation Curves of Training and CV Error as a function of
% Lambda Value

function [ThetasMat, min_ind] = lambdaValidationCurves(thetas, dims, X, y, trainSize)

% Useful Values:
m = size(X,1);

% Divide Into Training and Validation Set:
indsTrain = randperm(m, trainSize);
Xtrain = X(indsTrain,:);
ytrain = y(indsTrain,:);

indsVal = setdiff(1:m, indsTrain);
Xval = X(indsVal,:);
yval = y(indsVal,:);


% Initialize Lambda Values and Total Thetas Matrix:
lambdavec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100 300 1000 3000];
N = length(lambdavec);
ThetasMat = zeros(length(thetas), N);


% Initialize Error Vectors:
error_train = zeros(N,1);
error_val = zeros(N,1);

% Initialize Figure for Plotting Validation Curves:
figure(1); hold on;
plot(1:N, error_train, 1:N, error_val);
title(sprintf('Validation Curves for m_{train} = %d', length(indsTrain)));
xlabel('Lambda Vector Index Value');
ylabel('Error Value');
legend('Training Error', 'Validation Error');
pause on; pause(.5)

for n = 1:N
    fprintf('LAMBDA VALUE: %f\n', lambdavec(n));
    % Train Thetas for Specific Lambda Value and store in Theta Matrix:
    ThetasMat(:,n) = gradDescentTest(@(t)computeCost(t,dims, Xtrain,ytrain,lambdavec(n)), thetas, 400);
    
    % Compute Training and Validation Errors:
    error_train(n) = computeCost(ThetasMat(:,n), dims, Xtrain, ytrain, 0);
    error_val(n) = computeCost(ThetasMat(:,n), dims, Xval, yval, 0);
    
    % Plot Errors So Far:
    plot(1:n, error_train(1:n), 1:n, error_val(1:n));
    pause(.5);
end
pause off;

% Find the Lambda the Minimizes the Validation Error and Return the Respective Thetas:
[~, min_ind] = min(error_val);
fprintf('\nThe lambda value that minimizes the validation error is %f (index: %d)\n', lambdavec(min_ind), min_ind);
    
end
