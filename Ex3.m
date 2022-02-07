clear all
close all

n = 1000; % Generate Data .1
a1 = 1/2;
a2 = 1/3;
a3 = 1/6;

Zi = randsrc(n,1,[1 2 3; a1 a2 a3]); % Create labels

mu1 = [2 2];    % creating MU matrix (מטריצת התוחלת) and covarience matrix sig .2
mu2 = [-2 2];
mu3 = [0 -2];
mu = [mu1; mu2; mu3];
sig1 = [1.1 0; 0 1.1];
sig2 = [1.1 0; 0 1.1];
sig3 = [1.1 0; 0 1.1];

x1 = mvnrnd(mu1,sig1,n);   %creating 1000 2D data points from normal distribution .3
x2 = mvnrnd(mu2,sig2,n);
x3 = mvnrnd(mu3,sig3,n);
X = x1.*(Zi==1) + x2.*(Zi==2) + x3.*(Zi==3);

Nx1 = x1.*(Zi==1);
Nx1 = Nx1(Nx1 ~= 0);
Nx1 = reshape(Nx1',[],2);
Nx2 = x2.*(Zi==2);
Nx2 = Nx2(Nx2 ~= 0);
Nx2 = reshape(Nx2',[],2);
Nx3 = x3.*(Zi==3);
Nx3 = Nx3(Nx3 ~= 0);
Nx3 = reshape(Nx3',[],2);
Figure1 = figure('Units','centimeters','Position',[0 5 17.2 10]);
plot(Nx1(:,1),Nx1(:,2),'.')
hold on
plot(Nx2(:,1),Nx2(:,2),'.')
plot(Nx3(:,1),Nx3(:,2),'.')
plot(mu(:,1),mu(:,2),'p','MarkerEdgeColor','k')
hold off
title('Original labeled data');

%K-means algorithm

c1 = [2 0];  % Set the centroids:
c2 = [0 2];
c3 = [-2 0];
for j = 1:30
L1 = sum((X-c1).^2,2).^(1/2);
L2 = sum((X-c2).^2,2).^(1/2);
L3 = sum((X-c3).^2,2).^(1/2);
Lt = (min([L1';L2';L3']))';
Check1 = (Lt == L1);
Check2 = (Lt == L2);
Check3 = (Lt == L3);
c1 = (sum(Check1.*X))/sum(Check1);
c2 = (sum(Check2.*X))/sum(Check2);
c3 = (sum(Check3.*X))/sum(Check3);
labels = 1*Check1 + 2*Check2 + 3*Check3;
end
KM1 = min(sum((mu1-[c1; c2; c3]).^2,2).^(1/2));
KM2 = min(sum((mu2-[c1; c2; c3]).^2,2).^(1/2));
KM3 = min(sum((mu3-[c1; c2; c3]).^2,2).^(1/2));
error = sum(labels ~= Zi);
KMdeviation = KM1 + KM2 + KM3;
KMsucsess = (n - error)/n;

% Visualize the k-means algorithm
c = [c1; c2; c3];
K1 = X.*(labels==1);
K1 = K1(K1 ~= 0);
K1 = reshape(K1',[],2);
K2 = X.*(labels==2);
K2 = K2(K2 ~= 0);
K2 = reshape(K2',[],2);
K3 = X.*(labels==3);
K3 = K3(K3 ~= 0);
K3 = reshape(K3',[],2);
Figure4 = figure('Units','centimeters','Position',[17.2 5 17.2 10]);
plot(K1(:,1),K1(:,2),'.')
hold on
plot(K2(:,1),K2(:,2),'.')
plot(K3(:,1),K3(:,2),'.')
plot(c(:,1),c(:,2),'p','MarkerEdgeColor','k')
hold off
title('Data after K-means algorithm');

%Gaussian Mixture Model algorithm
p1 = 1/3;
p2 = 1/3;
p3 = 1/3;
GMMmu1 = [2 0];
GMMmu2 = [0 2];
GMMmu3 = [-2 0];
GMMsig1 = [1 0; 0 1];
GMMsig2 = [1 0; 0 1];
GMMsig3 = [1 0; 0 1];
 for j = 1:100
num1 = p1*mvnpdf(X,GMMmu1 ,GMMsig1);
num2 = p2*mvnpdf(X,GMMmu2,GMMsig2);
num3 = p3*mvnpdf(X,GMMmu3,GMMsig3);
num = num1 + num2 + num3;
wt1 = num1./num;
wt2 = num2./num;
wt3 = num3./num;
SumWt1 = sum(wt1);
SumWt2 = sum(wt2);
SumWt3 = sum(wt3);
p1 = SumWt1/n;
p2 = SumWt2/n;
p3 = SumWt3/n;
GMMmu1 = sum((wt1.*X))/SumWt1;
GMMmu2 = sum((wt2.*X))/SumWt2;
GMMmu3 = sum((wt3.*X))/SumWt3;
GMMsig1 = (((X-GMMmu1)')*(wt1.*(X-GMMmu1)))/SumWt1;
GMMsig2 = (((X-GMMmu2)')*(wt2.*(X-GMMmu2)))/SumWt2;
GMMsig3 = (((X-GMMmu3)')*(wt3.*(X-GMMmu3)))/SumWt3;
end
GMMlabels = (max([wt1';wt2';wt3']))';
GMMlabels = 1.*(wt1==GMMlabels) + 2.*(wt2==GMMlabels) + 3.*(wt3==GMMlabels);
gmmDev1 = min(sum((mu1-[GMMmu1; GMMmu2; GMMmu3]).^2,2).^(1/2));
gmmDev2 = min(sum((mu2-[GMMmu1; GMMmu2; GMMmu3]).^2,2).^(1/2));
gmmDev3 = min(sum((mu3-[GMMmu1; GMMmu2; GMMmu3]).^2,2).^(1/2));
GMMerror = sum(GMMlabels ~= Zi);
GMMdeviation = gmmDev1 + gmmDev2 + gmmDev3;
GMMsucsess = (n - GMMerror)/n;

% % Visualize the GMM algorithm
MU = [GMMmu1; GMMmu2; GMMmu3];
GMM1 = X.*(GMMlabels==1);
GMM1 = GMM1(GMM1 ~= 0);
GMM1 = reshape(GMM1',[],2);
GMM2 = X.*(GMMlabels==2);
GMM2 = GMM2(GMM2 ~= 0);
GMM2 = reshape(GMM2',[],2);
GMM3 = X.*(GMMlabels==3);
GMM3 = GMM3(GMM3 ~= 0);
GMM3 = reshape(GMM3',[],2);
Figure5 = figure('Units','centimeters','Position',[34.4 5 17.2 10]);
plot(GMM1(:,1),GMM1(:,2),'.')
hold on
plot(GMM2(:,1),GMM2(:,2),'.')
plot(GMM3(:,1),GMM3(:,2),'.')
plot(MU(:,1),MU(:,2),'p','MarkerEdgeColor','k')
hold off
title('Data after GMM algorithm');
