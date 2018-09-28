function Vasicek(k,r_bar,sigma,r)
% k is the speed of mean reversion
% r_bar is the risk-neutral long run mean of the short rate
% sigma is the volatility of the short rate
% Our model is dr(t)=k(r_bar-r(t))dt+sigma*dW, where dW is a Brownian
% motion.
% r is the current short rate at time t

if nargin<4
    r=.07;
    k=.3;
    r_bar=.08;
    sigma=.01;
end

step=1; % Yearly frequency
mat=[0:step:10]'; % Maturity matrix

%% Analytical solution
B=(1-exp(-k*mat))/k;
A=(r_bar-sigma^2/(2*k^2))*(B-mat)-(sigma^2*B.^2)/(4*k);
P=exp(A-B*r);

%% ODE solution
function dy=vasicek_ode(t,y) %#ok<INUSL>
dy(1,1)=(1/2)*sigma^2*y(2)^2-r_bar*k*y(2);
dy(2,1)=1-k*y(2);
end

[~,y]=ode45(@vasicek_ode,mat,[0 0]);
A_ode=y(:,1); B_ode=y(:,2);
P_ode=exp(A_ode-B_ode*r);

%% Simulation solution
delta_t=step/50;
no_sim=1000; no_per=max(mat)/delta_t;
rsim=zeros(no_sim,no_per); drsim=zeros(no_sim,no_per);
rsim(:,1)=r;
for j=2:no_per
    dW=randn(no_sim,1);
    drsim(:,j)=k*(r_bar-rsim(:,j-1))*delta_t+sigma*sqrt(delta_t)*dW;
    rsim(:,j)=rsim(:,j-1)+drsim(:,j);
end
P_sim=ones(length(mat),1);
for i=2:length(mat)
    P_sim(i)=mean(exp(-delta_t*sum(rsim(:,1:mat(i)/delta_t),2)));
end

%% Plots
figure
plot(mat,P,'b',mat,P_ode,'r o',mat,P_sim,'g *')
legend('Analytical Price','ODE Price','Simulation Price')
xlabel('Maturity')
ylabel('Price')
end