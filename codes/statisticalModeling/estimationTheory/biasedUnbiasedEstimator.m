clear all
close all

x = [-10:.1:10];
norm1 = normpdf(x,1,1);

norm2 = normpdf(x,0,4);

h = figure;
hold on
plot(x,norm1,'linewidth',2)
plot(x,norm2,'linewidth',2)

xlabel('\theta')
ylabel('f')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
set(gca,'box','on')
pbaspect([1 1 1])
legend('biased esti.','unbiased esti.')
saveTightFigure(h,'fisherInformationDemo.pdf')
