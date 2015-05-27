% gradDescent.m
% Implementation of Gradient Descent

function [X, fVal] = gradDescent(f, X0, MaxIter)

% Initialize parameters:
alpha = 0.01;
thresh = 1e-5;
X = X0;
[J, grad] = feval(f, X);
Jold = J + 1;

if MaxIter < 100
    MaxIter = 100;
end

% Initialize Iteration Counter:
t = 1;


while (t < MaxIter) && (abs(J-Jold) > thresh)  % As long as under MaxIter and J-Jold is larger than thresh 
    tic;
    
    % Adaptive Step Size Determination: -----------------------------------
    % Initialize Parameters for Adaptive Step Size While Loop Comparison:
    Xnew = X - alpha*grad;
    [Jnew, gradNew] = feval(f, Xnew);
    Jnewplus = feval(f, (Xnew - alpha*gradNew));
    while Jnewplus > Jnew
        % Decrease Alpha:
        alpha = .75 * alpha;
        fprintf('Here: Alpha = %f \t Future Change in Cost: %f\n', alpha, Jnewplus-Jnew);
        
        % Prep Next Loop:
        Xnew = X - alpha*grad;
        [Jnew, gradNew] = feval(f, Xnew);
        Jnewplus = feval(f, (Xnew - alpha*gradNew));
    end
    %----------------------------------------------------------------------
    
    % Perform Gradient Descent Step:
    Xold = X;                               % Necessary for Correct Output
    X = X - alpha*grad; 
    fprintf('%d: \t Cost is: %f \t Change in Cost is: %f \t Alpha: %f \t Iteration Time: %.3f sec\n', t, J, J-Jold, alpha, toc);
    
    % Prep Next Loop:
    Jold = J; 
    [J, grad] = feval(f, X);        % Perform One Step of Gradient Descent
    t = t+1;                        % Increment Iteration Counter
end

% Set Output Values:
X = Xold;
fVal = J;

% Print Warning If Necessary:
if t > MaxIter
    fprintf('Failure to Converge, Exceeded Maximum Iterations\n')
end

end