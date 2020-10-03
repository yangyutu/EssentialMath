clear all
close all

n = 500;
nu = 1;
T = mvtrnd([1 0; 0 1], nu, n);
U = tcdf(T,nu);
h=figure(1)
plot(U(:,1),U(:,2),'.');
title('\rho = 0');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'StudentTCopulaDemo1.pdf')

T = mvtrnd([1 .5; .5 1], nu, n);
U = tcdf(T,nu);
h=figure(2)
plot(U(:,1),U(:,2),'.');
title('\rho = 0.5');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'StudentTCopulaDemo2.pdf')

T = mvtrnd([1 0.8; 0.8 1], nu, n);
U = tcdf(T,nu);
h=figure(3)
plot(U(:,1),U(:,2),'.');
title('\rho = 0.8');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'StudentTCopulaDemo3.pdf')

T = mvtrnd([1 -.8; -.8 1], nu, n);
U = tcdf(T,nu);
h=figure(4)
plot(U(:,1),U(:,2),'.');
title('\rho = -0.8');
xlabel('U1');
ylabel('U2');
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
set(gca,'box','on')
saveTightFigure(h,'StudentTCopulaDemo4.pdf')