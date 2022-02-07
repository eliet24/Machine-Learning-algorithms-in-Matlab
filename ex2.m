clear all
close all
load('mnist_all.mat') 

 %//part D-Visualize some of the feature vectors, using reshape((28,28))
Test0Reshape = reshape(test0(1, :), [28, 28]);
Test1Reshape = reshape(test1(1, :), [28, 28]);
Test2Reshape = reshape(test2(1, :), [28, 28]);
Test3Reshape = reshape(test3(1, :), [28, 28]);
Test4Reshape = reshape(test4(1, :), [28, 28]);
Test5Reshape = reshape(test5(1, :), [28, 28]);
figure;
subplot(1,6,1);
imshow(Test0Reshape);
subplot(1,6,2);
imshow(Test1Reshape);
subplot(1,6,3);
imshow(Test2Reshape);
subplot(1,6,4);
imshow(Test3Reshape);
subplot(1,6,5);
imshow(Test4Reshape);
subplot(1,6,6);
imshow(Test5Reshape);

%part E-Train a binary Logistic Regression algorithm using the Gradient As-
%cent method described above. Train the machine to distinguish be-
%tween the digits 1 & 2.

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
 