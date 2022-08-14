function [AA,HH]=PUF(S)
AA = '';
HH='';
ref='10011011000010101101110111101011111111110011010000101101100001101000010010010110110110011111110101111001110110010110011010001000'
res = zeros(128,1);

for i = 1:128
x= int2str(i)   
Cx=strcat('Cycle',x);
fun=xlsread(S,Cx,'b:b')
dat=[fun(:,2)];

s=size(dat);
l=s(1);
b = zeros(l,1);
list=[];
BB='';
CC='';
OO='';

LL='';

    for j=1:l
        if (j<=l-1)
        
         X=fun(j,2)-fun((j+1),2);
        if X<0
            Y=X*(-1);
        else
            Y=X;
        end
        
        if Y>(3e-7)
           BB = strcat(BB,val2str9(fun(j,2)));
        LL =strcat(val2str9(fun(j,2)))
        OO =LL(3)
       % HH = strcat(HH,OO)
           %%%%%%%%
           
        end
         
       % CC = strcat(CC,twos_comp(fun(j,2)));
%       if (mod(j,2)==0)
%         OO=strcat(CC,BB);
%       else
%           OO=strcat(BB,CC);
%           
%       end
        
        %CC= strcat(dec2tc(fun(i,2),128))
        %De=val2de(fun(i,2))  
        %OO=dec2bin(De(i));
        % val2de(fun(i,2))
        end 
        
    end
  %  v = BB(1:128); 
  %  res(i) = hammingDist(ref, v);
    HH = strcat(HH,OO)
    AA = strcat(AA,BB)
    





end

%HH=''
%for h=1:4
       
        
         
      
         %  HH(h) = AA(h);
           
     
       

%end
HH
end


    
%for x = 1:100
%if rem(x,10)==1, tic, end
%De=ones(71,1);

    
%if rem(x,10)==0, toc, tic; end
%AA = strcat(AA,BB);
%end
%disp(BB)
%disp(CC)
% AA=zeros(l,1)
% for i=1:l
% b(i)=val2str9(fun(i,2));  
% end
% 
% AA=dec2bin(b)
