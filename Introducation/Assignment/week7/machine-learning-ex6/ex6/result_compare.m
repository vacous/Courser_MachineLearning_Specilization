function prediction_error = result_compare(parms,X,y,Xval,yval)

C=parms(1);
sigma=parms(2);



% Train SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
% calculate the prediction error 
prediction_error= mean(double(predictions ~= yval));


end