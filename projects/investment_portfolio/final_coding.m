clear;
T = readtable("E:\MATH 485\data.xlsx",'range','B2:J29');
%disp(T);
%The account value
COST = T(:,1);
NKE = T(:,2);
TWTR = T(:,3);
AAPL = T(:,4);
NFLX = T(:,5);
JNJ = T(:,6);
RGR = T(:,7);
GILD = T(:,8);
CLX = T(:,9);

%Calculate Return
prices = [COST,NKE,TWTR,AAPL,NFLX,JNJ,RGR,GILD,CLX];
%disp(prices);
%initialprice = [COST0,NKE0,TWTR0,AAPL0,NFLX0,JNJ0,RGR0,GILD0,CLX0];
%Returns = (prices-initialprice)./initialprice;
Returns = tick2ret(prices);
disp(Returns);
%Calculate summary statistics
Mean = mean(Returns);
%disp(Mean);
cov_Matrix = cov(Returns);
%disp(cov_Matrix);

%Setup for Optimization
muP = 0.04;
fun = @(x)x* cov_Matrix*x';
x0 = [0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111,0.111];
A = [];
b = [];
Aeq = [1,1,1,1,1,1,1,1,1;Mean];
beq = [1;muP];
lowb = [0,0,0,0,0,0,0,0,0];
uppb = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5];

[weights, varP] = fmincon(fun, x0, A, b, Aeq, beq, lowb, uppb);


 




