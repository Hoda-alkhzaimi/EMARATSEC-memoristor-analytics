function [V,C]=TD(S)
HC=zeros(9,1);

Hebac=zeros(128,1);
C=zeros(128,1);
De=zeros(9,1);
N="";
V=zeros(128,1);
Z=zeros(128,1);
for i = 1:128
x= int2str(i)   
Cx=strcat('Cycle',x);
fun=xlsread(S,Cx,'b:b');
dat=[fun(:,2)];

s=size(dat);
l=s(1);
b = zeros(l,1);

HC=zeros(9,1);
    for j=1:l
        if (j<=l-1)
        
         X=fun(j,2)-fun((j+1),2);
        if X>(3e-7)
            V(i)=fun(j,3)
            %C(i)=(val2de(fun((j+1),2)))
             C(i)=(fun((j+1),2))
%            De=C(i)
%             a = num2str(De)
%    for k = 1:size(a,2)
%        if (k>5)
%           HC =strcat(HC,a(k))
%  %  HC(k) = HC(k) + a(k)
%        end
%    end

% Hebac(i)=str2num(HC(1,:))
%Hebac(i)=str2num(HC(1))
          % C(i)=strcat(val2str9(fun(j,2)))
            break
            
        end
        end
    end
    
    if (V(i)<1.75)
                Z(i)=0;
            else Z(i)=1;
            end
    X=[V,C]
end