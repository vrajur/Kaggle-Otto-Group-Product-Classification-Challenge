% NNPredict.m
% Make Prediction for Given Input Examples

function [y_out, p] = NNPredict(trained_thetas, dims, X)

% Unwrap Thetas into Matrix Variables:
cut = dims(2)*(dims(1)+1);                                % Numel in theta1
theta1 = reshape(trained_thetas(1:cut), [dims(2), dims(1)+1]);        % s2 x (s1+1)
theta2 = reshape(trained_thetas(cut+1:end), [dims(3), dims(2)+1]);    % s3 x (s2+1)


% Feedforward to Generate Predictions:
y_out = feedForward(theta1, theta2, X);


% Calulate Probabilities:
p = zeros(size(y_out));
for i = 1:size(p,1)
    totalRowOutput = sum(y_out(i,:));
    for j = 1:size(p,2)
        p(i,j) = y_out(i,j)/totalRowOutput;
    end
end


end