clear all
close all


ta = 1;
tb = 10;

xa = 1;
xb = 3;

h = figure(1)
plot(ta,xa,'square','color','red','linewidth',2);
hold on
plot(tb,xb,'square','color','red','linewidth',2);

t1 = 5;
mu = [ta t1]*inv([ta ta;ta tb])*[xa xb]';
sigma = t1 - [ta t1]*inv([ta ta;ta tb])*[ta t1]';
for i = 1:10

x1 = normrnd(mu,sigma);
    plot(t1,x1,'o','color','blue','linewidth',2);

end
xlabel('time t')
ylabel('X(t)')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'BrownianInterpolationDemo1.pdf')


ta = 1;
tb = 10;

xa = 1;
xb = 3;

figure(2)
plot(ta,xa,'o');
hold on
plot(tb,xb,'o');
tset=[2,3,4,5,6,7,8,9];
tall = [ta tset tb];
for i = 1:5
    
xa = 1;
ta = 1;
xtraj = [];
xtraj = [xtraj xa];
for j=1:length(tset)
t1 = tset(j);
mu = [ta t1]*inv([ta ta;ta tb])*[xa xb]';
sigma = t1 - [ta t1]*inv([ta ta;ta tb])*[ta t1]';


x1 = normrnd(mu,sigma);
xtraj = [xtraj x1];
xa = x1;
ta = t1;
end
xtraj = [xtraj xb];

h = figure(2)
hold on
plot(tall,xtraj,'-o','linewidth',2);
end

xlabel('time t')
ylabel('X(t)')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);

pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'BrownianInterpolationDemo2.pdf')
