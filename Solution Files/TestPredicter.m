% TestPredicter.m
% Load the Test Set and Write the Class Predictions to a CSV File:

function [y, p] = TestPredicter(thetas, dims)

% Load the Test Set:
S = load('test.mat');
Xtest = S.('X');
ids = S.('ids');

% Make Predictions and Create Output Array:
[y, p] = NNPredict(thetas, dims, Xtest);
A = [ids, p];


% Write Predictions to CSV File:
f = fopen('Predictions.csv', 'w');

% Write Header Line:
fprintf(f, 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n');

% Write Predictions:
fprintf(f, '%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n', A');

% Close CSV File:
fclose(f);


end