function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure



% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


%positive index and negative index
i_p=find(y==1);
i_n=find(y==0);

figure()
hold on
plot(X(i_p,1),X(i_p,2),'b+')
plot(X(i_n,1),X(i_n,2),'ro')
title 'Exam 1 and 2 score'
hold off



% =========================================================================


end
