%[X,y]=TD('Trainingdata.xls')
data=xlsread('training1664.xls','Sheet1','b:b')
MM=data(:,[1,2])
H=data(:,[1,2])
%M=data(:,3)
J=data(:,3)

% plotTD(H,M)
% xlabel('bit 1')
% ylabel('bit 2')
% legend('+ binary bit 1', 'O binary bit 0')

plotTD(H,J)
xlabel('Voltage')
ylabel('Current')
legend('Binary bit 1', 'Binary bit 0')


[m,n]=size(H);
H=[ones(m,1) H];

%intialize intial parameters;
initial_theta=zeros(n+1, 1);
% compute cost and gradient
[cost,grad]=CostFunction(initial_theta,H,J);
fprintf('Cost at intial theta(zeros):%f\n',cost);
disp('Gradient at initial theta(zeros):');disp(grad);
%set options for fminunc
options=optimoptions(@fminunc, 'Algorithm', 'Quasi-Newton', 'GradObj', 'on', 'MaxIter', 400);
%run fminunc to obtain the optimal theta
[theta,cost]=fminunc(@(t)(CostFunction(t, H,J)),initial_theta, options);
plotDecisionBoundary(theta, H, J);
hold on 
xlabel('bit 1')
ylabel('bit 2')
legend('Binary 1', 'Binary 0');
hold off

%prop=sigmoid([1 2.5 3.5]*theta);
p=predict(theta,H)
