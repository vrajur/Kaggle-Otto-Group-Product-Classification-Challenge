% initializer.m
% Initialize the Parameters:

% Prepare Workspace:
close, clear all, clc;

fprintf('Initializing Training Set and Neural Net...');

% Load training data and Initial Matrices:
S = load('train.mat');
X = S.('X');
y = S.('y');


% Define Useful Variables:
[m, n] = size(X);               % 61878 x 93
k = size(y,2);                  % number of output classes: k = 9


% Randomly Order the X and y Matrices:
randInds = randperm(m);
X = X(randInds,:);
y = y(randInds,:);


% Define a 1-hidden layer Neural Network:
s1 = n;                         % 93 Input Features
s2 = 51;                        % 93 non-bias hidden layer nodes (prunable)
s3 = k;                         % 9 output nodes


% Initialize Thetas (Random Values to Break Symmetry):
eps1 = sqrt(6)/sqrt(s1+s2);                 % magnitude of random init.
theta1 = eps1*(2*rand(s2, s1+1)-1);         % s2 x (s1+1)

eps2 = sqrt(6)/sqrt(s2+s3);                 % magnitude of random init.
theta2 = eps2*(2*rand(s3, s2+1)-1);         % s3 x (s2+1)

% Unroll Thetas into Wrapper Variable:
thetas = [theta1(:); theta2(:)];            % column vector of thetas
dims = [s1, s2, s3];                        % relevant dimensions for thetas

fprintf('\tInitialized!\n\n');
