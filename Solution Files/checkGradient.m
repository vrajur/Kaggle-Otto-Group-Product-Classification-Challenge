% checkGradient.m
% Test the accuracy of gradient computation

function checkGradient(n, thetas, dims, X, y, lambda)

m = size(X,1);
inds = randperm(m,n);
xt = X(inds,:);
yt = y(inds,:);
[~, ~, g] = computeCost(thetas,dims, xt, yt, 1);
numericGradient(g, thetas, dims, xt, yt, 1);

end