%part G-
xt = double(test1);
y = xt*wi';
sig = sigmf(y, [1 0]);
xt = double(test2);
y = xt*wi';
sig = sigmf(y, [1 0]);
ErrorsNum = sum(sig >= .5)+ sum(sig <= .5);
successRate = (1-(ErrorsNum/NTest))*100