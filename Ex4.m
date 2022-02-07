clear all
close all
load('facesData.mat') 
train = zeros(120,1024);
labelTrain = zeros(120,1);
test = zeros(45,1024);
labelTest = zeros(4,1);
for n = 1:15
train(1+8*(n-1):8*n,:) = faces(1+11*(n-1):11*n-3,:);
labelTrain(1+8*(n-1):8*n,:) = labeles(1+11*(n-1):11*n-3,:);
test(1+3*(n-1):3*n,:) = faces(9+11*(n-1):11*n,:);
labelTest(1+3*(n-1):3*n,:) = labeles(9+11*(n-1):11*n,:);
end
mean_train = mean(train);
new_train = train - mean_train;
mean_test = mean(test);
new_test = test - mean_test;
cov_train = (1/120)*((new_train')*new_train);
[V,D] = eig(cov_train);
V = fliplr(V);
Success_rate = zeros(1,111);
j = 1;
for i = 1:111
Train = new_train * V(:,1:i);
Test = new_test * V(:,1:i);
guess_vector = zeros(45,1);
for n = 1:45
[distance,index] = min(sum((Test(n,:)-Train).^2,2).^(1/2));
guess_vector(n,:) = index;
end
label_guess_Vector = labelTrain(guess_vector);
Error = sum(label_guess_Vector ~= labelTest);
Success_rate(:,j) = (45 - Error)/45;
j = j + 1;
end
%% Visualization
img0 = zeros(32*15,32*11);
for n = 1:15
for k = 1:11
img0(1+32*(n-1):32*n,1+32*(k-1):32*k)= reshape(faces(k+11*(n-1),:),32,32);
end
end
Figure1 = figure('Units','pixels','Position',[0 0 480 352]);
imshow(img0,[])
img1 = zeros(32*15,32*8);
for n = 1:15
for k = 1:8
img1(1+32*(n-1):32*n,1+32*(k-1):32*k)= reshape(new_train(k+8*(n-1),:),32,32);
end
end
Figure2 = figure('Units','pixels','Position',[480 0 480 352]);
imshow(img1,[])
Figure3 = figure('Units','centimeters','Position',[0 5 17.2 10]);
plot(1:111,Success_rate,'LineWidth',2)
title('Success rate \propto Number of components');
xlabel('Number of components [#]')
ylabel('Success rate')
axis tight
