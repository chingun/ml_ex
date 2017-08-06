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
    n = length(theta);
    
    x = (1 - lambda/m);
    
    for i = 1:m
        if(y(i))
            J = J - (1/(m)*log(sigmoid(transpose(theta)*transpose(X(i,:)))));
        else
            J = J - (1/(m)*log(1 - sigmoid(transpose(theta)*transpose(X(i,:)))));
        end
        
        grad = grad + transpose(1/m * (sigmoid(transpose(theta)*transpose(X(i,:))) - y(i)) .* X(i,:));
    end
    
    for j = 2:n
         J = J + lambda/(2*m) * (theta(j))^2;
         
         if(j < 2)
                grad(j) = grad(j);
         else
                grad(j) = grad(j) + lambda/m*theta(j);
         end
    end

% =============================================================

end
