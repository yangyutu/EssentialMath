clear all
close all

%% demo for normal data
mu1 = [1 2];
sigma1 = [2 0; 0 .5];
mu2 = [-3 -5];
sigma2 = [1 0; 0 1];
rng(1); % For reproducibility
X = [mvnrnd(mu1,sigma1,1000);
     mvnrnd(mu2,sigma2,1000)];
 
 scatter(X(:,1),X(:,2),10,'.') % Scatter plot with points of size 10
title('Simulated Data')

options = statset('Display','final');
gm = fitgmdist(X,2,'Options',options)

gmPDF = @(x,y)pdf(gm,[x y]);
hold on
h = ezcontour(gmPDF,[-8 6],[-8 6]);
title('Simulated Data and Contour lines of pdf');



