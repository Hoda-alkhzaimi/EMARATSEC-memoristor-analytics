function g=sigmoid(z)
g=zeros(size(z));
[m,n]=size(z);

for i=1:m
    for j=1:n
        
        g(i,j)=1/(1+exp(-z(i,j)))
        
    end
    
end


end