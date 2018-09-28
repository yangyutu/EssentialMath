clear all
close all
%% Estimate and Plot Factor Loadings
%%
% Load the sample data.

% Copyright 2015 The MathWorks, Inc.

load carbig
%%
% Define the variable matrix.
X = [Acceleration Displacement Horsepower MPG Weight]; 
X = X(all(~isnan(X),2),:);
%%
% Estimate the factor loadings using a minimum mean squared error
% prediction for a factor analysis with two common factors.
% Lambda is the factor loading matrix 5 by 2
% F is the estimated factors via PCA
[Lambda,Psi,T,stats,F] = factoran(X,2,'scores','regression');
inv(T'*T);   % Estimated correlation matrix of F, == eye(2)
Lambda*Lambda' + diag(Psi); % Estimated correlation matrix
Lambda*inv(T);              % Unrotate the loadings
F*T';                       % Unrotate the factor scores
%%
% Create biplot of two factors.
biplot(Lambda,'LineWidth',2,'MarkerSize',20)
%%
% Estimate the factor loadings using the covariance (or correlation)
% matrix.
[Lambda,Psi,T] = factoran(cov(X),2,'xtype','cov')
% [Lambda,Psi,T] = factoran(corrcoef(X),2,'xtype','cov')
%%
% Although the estimates are the same, the use of a covariance matrix
% rather than raw data doesn't let you request scores or significance
% level.
%%
% Use promax rotation.
[Lambda,Psi,T,stats,F] = factoran(X,2,'rotate','promax',...
                                      'powerpm',4);
inv(T'*T)                            % Estimated correlation of F, 
                                     % no longer eye(2)
Lambda*inv(T'*T)*Lambda'+diag(Psi)   % Estimated correlation of X
%%
% Plot the unrotated variables with oblique axes superimposed.
invT = inv(T);
Lambda0 = Lambda*invT;
figure()
line([-invT(1,1) invT(1,1) NaN -invT(2,1) invT(2,1)], ...
     [-invT(1,2) invT(1,2) NaN -invT(2,2) invT(2,2)], ...
     'Color','r','linewidth',2)
grid on
hold on
biplot(Lambda0,'LineWidth',2,'MarkerSize',20)       
xlabel('Loadings for unrotated Factor 1')
ylabel('Loadings for unrotated Factor 2')
%%
% Plot the rotated variables against the oblique axes.
figure()
biplot(Lambda,'LineWidth',2,'MarkerSize',20)


