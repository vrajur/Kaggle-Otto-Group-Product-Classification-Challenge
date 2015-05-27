% gradDescent.m
% Implementation of Gradient Descent

function [X, fVal] = gradDescentTimed(f, X0, MaxIter)

% disp(feval(f, X0));

% Initialize parameters:
alpha = 0.0001;
thresh = 1e-9;
Xnew = X0;
Xold = X0 + 1;

if MaxIter < 100
    MaxIter = 100;
end

% Initialize Iteration Counter:
t = 1;
tic

while (t < MaxIter) && (abs(mean(Xnew - Xold)) > thresh)

    [J, grad] = feval(f, Xnew);     % Evaluate Function Value and Gradient
    fprintf('Iteration %d: \t Cost is: %f \t Estimated Completion Time: ', t, J);
    Xold = Xnew;    
    Xnew = Xold - alpha*grad;       % Perform One Step of Gradient Descent
    
    if t == 1
        timeCount = toc;                % Time for One Iteration
        maxTimeLeft = timeCount*MaxIter;   % Estimate Total Time
    end
    
    % Calculate and Print the Time Left:
    timeLeft = maxTimeLeft - timeCount*t;  
    h = floor(timeLeft/3600); timeLeft = timeLeft - 3600*h;
    m = floor(timeLeft/60); timeLeft = timeLeft - 60*m;
    s = floor(timeLeft); 
    fprintf('%d hr %d min %d sec left\n', h, m, s);
        
    t = t+1;                        % Increment Iteration Counter
end

% Set Output Values:
X = Xnew;
fVal = J;

% Print Warning If Necessary:
if t > MaxIter
    fprintf('Failure to Converge, Exceeded Maximum Iterations\n')
end

end