clear all
close all

x = [0:.1:10];
e = randn(1,size(x,2)).*x/5;
y = x + e;

h = figure;
hold on
plot(x,y,'.','markersize',10)
plot(x,x,'-','linewidth',2)

xlabel('x')
ylabel('y')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
set(gca,'box','on')
pbaspect([1 1 1])
saveTightFigure(h,'heteroscedasiticityDemo.pdf')
