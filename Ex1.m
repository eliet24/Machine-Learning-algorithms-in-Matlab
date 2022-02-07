n = 1000;
m = 500;
s = 100;
error = zeros(1,s);
X = rand(n,m);
beta = 500*ones(m,1);
for sigma = 1:s
e = normrnd(0,sigma,n,1);
Y = X*beta + e;
bTag = (((X.')*X)^-1)*(X.')*Y;
error(sigma) = mean(abs(beta - bTag));
end
sigma = 1:s;
plot(sigma,error,'LineWidth',2)
title('Error \propto Sigma');
xlabel('Sigma')
ylabel('Error')
axis tight
