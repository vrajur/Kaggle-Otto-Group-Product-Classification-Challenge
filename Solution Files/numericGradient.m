% numericGradient.m
% Compute the Gradient Numerically for Comparison

function gradNum = numericGradient(grad, thetas, dims, X, y, lambda)

N = length(thetas);
eps = 1e-4;
perturb = zeros(size(thetas));
gradNum = zeros(size(thetas));

for i = 1:N
    fprintf('Iteration: %d of %d\n', i, N);
    perturb(i) = eps;
    cost_plus = computeCost(thetas+perturb, dims, X, y, lambda);
    cost_minus = computeCost(thetas-perturb, dims, X, y, lambda);
    gradNum(i) = 0.5/eps * (cost_plus - cost_minus);
    perturb(i) = 0;
end

fprintf('\nNumerical:\tBackProp:\n');
disp([gradNum(1:15), grad(1:15)]);

diff = norm(gradNum-grad)/norm(gradNum+grad);
fprintf('Norm Difference Between Gradients: %g\n', diff);
end