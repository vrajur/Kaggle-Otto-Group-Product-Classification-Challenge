% feedForward.m
% Feeds Input Forward Through Supplied 3-Layer Neural Network

function [a3, a2, a1] = feedForward(theta1,theta2, X)

% Useful Values:
m = size(X, 1);

% Feed Forward:
% NOTE: IMPLICIT ASSUMPTION X>0!!! --> GONE NOW
a1 = [ones(m,1), X];                % m x (s1+1)

z2 = a1*theta1';                    % m x s2
a2 = [ones(m,1), sigmoid(z2)];      % m x (s2+1)

z3 = a2*theta2';                    % m x s3
a3 = sigmoid(z3);                   % m x s3



end