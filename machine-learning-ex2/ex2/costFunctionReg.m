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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% the cost function
htheta = sigmoid(X*theta);

% the penalty, exclude the first theta(1) which is theta0 in eq.
theta1 = [0; theta(2:size(theta), :)];

p = lambda*(theta1'*theta1)/(2*m);

% J = sum((-y .*log(htheta) - (1-y) .*log(1-htheta)))/m + p;  % <-- another exp. tested
J = ((-y)'*log(htheta) - (1-y)'*log(1-htheta))/m + p;

%grad = sum((htheta - y) .* X(:, i))/m + lambda*theta1/m;   %error: attempted to use a complex scalar as an index (forgot to initialize i or j?)
grad = (X'*(htheta - y)+lambda*theta1)/m;

% =============================================================

end
