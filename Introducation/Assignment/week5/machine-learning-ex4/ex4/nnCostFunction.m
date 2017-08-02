function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X=[ones(size(X,1),1),X];


z_2=X*Theta1';
hidden_layer=sigmoid(z_2);
hidden_layer=[ones(size(hidden_layer,1),1),hidden_layer];

output_layer=hidden_layer*Theta2';

h_theta=sigmoid(output_layer);

% convert y into logic array 

y_logic=zeros(size(y,1),range(y)+1);
for i=1:size(y,1)
    y_logic(i,y(i))=1;
end
    
% convert h_theta into logic array
% 
% [~,i_theta]=max(h_theta,[],2);
% 
% h_theta_logic=zeros(size(y_logic));
% for i=1:length(i_theta);
%     h_theta_logic(i,i_theta(i))=1;
% end



sum_m=0;

for i=1:size(y_logic,1)
    
    y_k=y_logic(i,:);
    h_theta_k=h_theta(i,:);
    
    sum_k=1/m*sum((-y_k.*log(h_theta_k)-(1-y_k).*log(1-h_theta_k)));
    
    sum_m=sum_m+sum_k;
end

% layer1-2 regularization
sum_j1=0;
for i=1:size(Theta1,1)
    
    sum_k=sum(Theta1(i,:).^2.*[0,ones(1,size(Theta1,2)-1)]);
    
    sum_j1=sum_j1+sum_k;
end

%layer2-3 regularization
sum_j2=0;
for i=1:size(Theta2,1)
    
    sum_k=sum(Theta2(i,:).^2.*[0,ones(1,size(Theta2,2)-1)]);
    
    sum_j2=sum_j2+sum_k;
end

J=sum_m+lambda/(2*m)*(sum_j1+sum_j2);
%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
delta_3=h_theta-y_logic;
delta_2=delta_3*Theta2.*sigmoidGradient([ones(size(z_2,1),1),z_2]);
delta_2=delta_2(:,2:end);

Delta_2=delta_3'*hidden_layer;
Delta_1=delta_2'*X;

Theta1_grad=1/m*Delta_1;
Theta2_grad=1/m*Delta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad=Theta1_grad+lambda/m*Theta1...
    .*[zeros(size(Theta1,1),1),ones(size(Theta1,1),size(Theta1,2)-1)];
Theta2_grad=Theta2_grad+lambda/m*Theta2...
    .*[zeros(size(Theta2,1),1),ones(size(Theta2,1),size(Theta2,2)-1)];












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
