function [J, grad]=CostFunction(theta, X, y)
m=length(y)
J=0;
grad=zeros(size(theta));

h=sigmoid(X*theta)
J=(-1/m)*sum((y.*log(h)) +((1-y).*log(1-h)));

for j=1:size(grad)
    grad(j)=(1/m)*sum((h-y).*X(:,j));
end
end
