function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% theta=initial_theta;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h_theta=sigmoid(X*theta);
logmulti=ones(length(theta),1);
logmulti(1)=0;
J=1/m*sum(-y.*log(h_theta)-(1-y).*log(1-h_theta))+lambda/(2*m)*sum(theta.^2.*logmulti);

for j=1:size(X,2)
    if j==1
        grad(j)=1/m*sum((h_theta-y).*X(:,j));
    else
        grad(j)=1/m*sum((h_theta-y).*X(:,j))+lambda/m*theta(j);
    end
end






% =============================================================

end
