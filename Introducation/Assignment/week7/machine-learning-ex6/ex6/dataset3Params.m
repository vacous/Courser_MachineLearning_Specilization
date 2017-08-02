function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C=1; % C
sigma=0.3; % sigma 



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

% run fmincon to find the 
% wrapper=@(parms)result_compare(parms,X,y,Xval,yval);
% 
% optimized_parms = fmincon(wrapper,ini_parms,[],[],[],[],[0 0.1],[30 0.3],[]);
% 
% C=optimized_parms(1);
% sigma=optimized_parms(2);
value_pool=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

for i=1:length(value_pool) % C
    for j=1:length(value_pool) % sigma
        
        C_temp=value_pool(i);
        sigma_temp=value_pool(j);
        
        model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
        predictions = svmPredict(model, Xval);
        prediction_error_temp= mean(double(predictions ~= yval));
        
        result_array{i,j}=[prediction_error_temp,C_temp,sigma_temp];
    end
end

% convert the result_array into vector 

result_matrix=cell2mat(reshape(result_array,64,1));

% find the index which has the min prediction error
[~,I]=min(result_matrix(:,1),[],1);

C=result_matrix(I,2);
sigma=result_matrix(I,3);

% =========================================================================

end
