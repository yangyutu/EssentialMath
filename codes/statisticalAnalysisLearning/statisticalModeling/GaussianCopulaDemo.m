clear all
close all

n = 500;


Z = mvnrnd([0 0], [1 0; 0 1], n);
U = normcdf(Z,0,1);
h=figure(1)
plot(U(:,1),U(:,2),'.');
title('\rho = 0');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'GaussianCopulaDemo1.pdf')

Z = mvnrnd([0 0], [1 .5; .5 1], n);
U = normcdf(Z,0,1);
h=figure(2)
plot(U(:,1),U(:,2),'.');
title('\rho = 0.5');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'GaussianCopulaDemo2.pdf')

Z = mvnrnd([0 0], [1 0.8; 0.8 1], n);
U = normcdf(Z,0,1);
h=figure(3)
plot(U(:,1),U(:,2),'.');
title('\rho = 0.8');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'GaussianCopulaDemo3.pdf')

Z = mvnrnd([0 0], [1 -.8; -.8 1], n);
U = normcdf(Z,0,1);
h=figure(4)

plot(U(:,1),U(:,2),'.');
title('\rho = -0.8');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'GaussianCopulaDemo4.pdf')