function p= predict (theta, X)
m=size(X,1);
p=zeros(m,1);
p=sigmoid(X*theta)
%prop=sigmoid([1 2]*theta);


for i=1:m
    if (p(i)>=0.5)
        p(i)=1
    else
     p(i)=0;
    end
    
end



end