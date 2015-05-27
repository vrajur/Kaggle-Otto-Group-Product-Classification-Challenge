% computeCost.m
% Evaluate the Cost Function and Calculate the Gradient as well as the
% Logloss Test Metric as defined by the Kaggle Competition

function [J, L, grad] = computeCost(thetas, dims, X, y, lambda)

% Useful Values:
m = size(X, 1);

% Initial Theta Variables:
theta1 = zeros(dims(2), dims(1)+1);           % s2 x (s1+1)
theta2 = zeros(dims(3), dims(2)+1);           % s3 x (s2+1)

theta1_grad = zeros(size(theta1));          % same as theta1
theta2_grad = zeros(size(theta2));          % same as theta2

% Unwrap Thetas into Matrix Variables:
cut = dims(2)*(dims(1)+1);                                % Numel in theta1
theta1 = reshape(thetas(1:cut), [dims(2), dims(1)+1]);        % s2 x (s1+1)
theta2 = reshape(thetas(cut+1:end), [dims(3), dims(2)+1]);    % s3 x (s2+1)


% Initialize Output Variables: (commented for efficiency)
% J = 0;                          % Cost Function Evaluation
% L = 0;                          % Logloss Test Metric
% grad = zeros(size(thetas));     % Gradient of Thetas


%============================ Compute Cost: ===============================
% Feed Forward:
[a3, a2, a1] = feedForward(theta1, theta2, X);

% Define Temporary Thetas for Regularized Evaluation:
temp_theta1 = theta1; temp_theta1(:,1) = 0;
temp_theta2 = theta2; temp_theta2(:,1) = 0;
temp_theta = [temp_theta1(:); temp_theta2(:)];      % Combined Theta matrix

% Cost Evaluation:
J = 1/m .* sum(sum(-y.*log(a3) - (1-y).*log(1-a3))) ...
    + lambda/(2*m) * sum(temp_theta.^2);

%============================ Compute Score: ==============================
% Evaluate Probabilities:
p = zeros(size(y));
for i = 1:size(p,1)
    totalRowOutput = sum(a3(i,:));
    for j = 1:size(p,2)
        p(i,j) = a3(i,j)/totalRowOutput;
    end
end

L = -1/m * sum(sum(y'*log(p)));

%======================== Evaluate the Gradient: ==========================
% BackPropagation:
for t = 1:m
   % Feedforward Training Example:
   a3_t = a3(t,:)';                     % s3 x 1
   
   % Initial Delta:
   delta3 = a3_t - y(t,:)';             % s3 x 1
   
   % Hidden Layer Delta:
   a2_t = a2(t,:)';                                 % (s2+1) x 1
   delta2 = theta2'*delta3 .* a2_t.*(1-a2_t);       % (s2+1) x 1
   
   % Accumulate Delta Sums:
   a1_t = a1(t,:);                     % 1 x (s1+1) 
   theta1_grad = theta1_grad + delta2(2:end)*a1_t;  % s2 x (s1+1)
   theta2_grad = theta2_grad + delta3*a2_t';        % s3 x (s2+1)
end

% Prepare Grad for Output:
theta1_grad = 1/m * (theta1_grad + lambda*temp_theta1);
theta2_grad = 1/m * (theta2_grad + lambda*temp_theta2);

grad = [theta1_grad(:); theta2_grad(:)];


% Check Gradient:
% checkGradient(grad,thetas,dims,X,y,lambda);


end