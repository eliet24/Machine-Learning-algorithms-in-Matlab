load('mnist_all.mat') 

SizeRow = size(train0, 2);
SizeTrain1 = size(train1, 1);
SizeTrain2 = size(train2, 1);
SizeTest1 = size(test1, 1);
SizeTest2 = size(test2, 1);
NTrain = SizeTrain1 + SizeTrain2;
NTest =SizeTest1 + SizeTest2;
NumOfreg = 50;
wi = ones(1, SizeRow) * 0.0005;
t = 0.0001;
costFuncHistogram = zeros(1, NumOfreg);
for i = 1:NumOfreg
xt = double(train1);
wixt = xt*wi';
sig = sigmf(wixt, [1 0]);
costFunc = sum((0-sig).*xt);
xt = double(train2);
wixt = xt*wi';
sig1 = sigmf(wixt, [1 0]);
costFunc = costFunc + sum((1-sig1).*xt);
wi = wi + (t/NTrain) * costFunc;
costFuncHist(i) = (sum(log(sig)) + sum(log(sig1)))/NTrain;
end

%part F-Print the Cost Function lw at each iteration of the optimization pro-
%cedure and verify it increases.
 figure;
plot(1:NumOfreg, costFuncHist);
title('$Cost function \propto Interation$', 'Interpreter', 'latex');
xlabel('$Interation number$', 'Interpreter', 'latex');
ylabel('$Cost function$', 'Interpreter', 'latex');

%part G-
xt = double(test1);
y = xt*wi';
sig = sigmf(y, [1 0]);
ErrorsNum = sum(sig >= .5);
xt = double(test2);
y = xt*wi';
sig = sigmf(y, [1 0]);
ErrorsNum2 = sum(sig <= .5);
numberOfErrors = ErrorsNum + ErrorsNum2;
successRate = (1-(numberOfErrors/NTest))*100