function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.03;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma = [0.01 0.03 0.1 0.3 1 3 10 30];

res = [];
for i = C
  for j = sigma
    fprintf('Train SVM with following parameters:\n');
    fprintf(['C = \t\t%f\n'], i);
    fprintf(['sigma = \t%f'], j);
    
    % Train the SVM
    model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    res = [error i j; res];
    fprintf(['Error = \t\t%f\n'], error);
  end
end

[val, ind] = min(res(:, 1));

result = res(ind, :);

C = result(1, 2)
sigma = result(1, 3)
% =========================================================================

end
