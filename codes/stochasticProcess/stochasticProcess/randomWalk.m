clear all
close all




T = [0, 0, 1, 1, 2, 2, 3];
X1 = [0, 1, 1, 2, 2, 1, 1];
T2 = [0, 1, 1, 2, 2, 3, 3];
X2 = [0, -1, -1, 0, 0, -1, -1];
h = figure(1)
plot(T,X1,'linewidth',2,'color','red');
hold on
plot(T,X2,'linewidth',2,'color','blue');
ylim([-2 3]);
xlim([0.0 5]);
xlabel('time')
ylabel('position')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
saveTightFigure(h,'randomWalkPaths.pdf')