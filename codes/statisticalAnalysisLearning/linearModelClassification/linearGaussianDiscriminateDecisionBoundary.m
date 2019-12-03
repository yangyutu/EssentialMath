clear all
close all


mu1 = [-3,-5]; %// data
sigma = [5 1; 1 5]; %// data
x = -10:.1:10; %// x axis
y = -10:.1:10; %// y axis

[X Y] = meshgrid(x,y); %// all combinations of x, y
Z1 = mvnpdf([X(:) Y(:)],mu1,sigma); %// compute Gaussian pdf
Z1 = reshape(Z1,size(X)); %// put into same size as X, Y

mu2 = [3, 3];
Z2 = mvnpdf([X(:) Y(:)],mu2,sigma); %// compute Gaussian pdf
Z2 = reshape(Z2,size(X)); %// put into same size as X, Y


mu3 = [4, -5];
Z3 = mvnpdf([X(:) Y(:)],mu3,sigma); %// compute Gaussian pdf
Z3 = reshape(Z3,size(X)); %// put into same size as X, Y

h=figure(1)
contour(X,Y,Z1+Z2), axis equal  %// contour plot; set same scale for x and y...
xlabel('x')
ylabel('y')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
saveTightFigure(h,'linearGaussianDiscriminateDecisionBoundaryDemo2DOne.png')

h=figure(2)
surf(X,Y,Z1+Z2,'linestyle','none') %// ... or 3D plot
xlabel('x')
ylabel('y')
zlabel('z')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
saveTightFigure(h,'linearGaussianDiscriminateDecisionBoundaryDemo3DOne.png')

h=figure(3)
contour(X,Y,Z1+Z2 + Z3), axis equal  %// contour plot; set same scale for x and y...
xlabel('x')
ylabel('y')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
pbaspect([1 1 1])
saveTightFigure(h,'linearGaussianDiscriminateDecisionBoundaryDemo2DTwo.png')


h=figure(4)
surf(X,Y,Z1+Z2+Z3,'linestyle','none') %// ... or 3D plot
xlabel('x')
ylabel('y')
zlabel('z')
set(gca,'linewidth',2,'fontsize',15,'fontweight','bold','plotboxaspectratiomode','auto','xminortick','on','yminortick','on','TickLength',[0.04;0.02]);
saveTightFigure(h,'linearGaussianDiscriminateDecisionBoundaryDemo3DTwo.png')