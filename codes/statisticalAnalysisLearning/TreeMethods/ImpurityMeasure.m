clear all
close all

p = 0: 0.01: 1;

entropy = -p.*log(p) - (1-p).*log(1-p);

Gini = 1 - p.^2 - (1- p).^2;

ClassifyError = 1 - max(p, 1 - p);

figure(1)

plot(p, entropy, p, Gini, p, ClassifyError,'linewidth',2);
xlim([0 1])
ylim([0 1])
xlabel('p')
ylabel('impurity measure')
legend('entropy','Gini','class. err.')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])