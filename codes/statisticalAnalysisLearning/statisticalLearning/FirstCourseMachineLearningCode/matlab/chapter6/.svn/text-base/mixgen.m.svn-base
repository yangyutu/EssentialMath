%% mixgen.m
% From A First Course in Machine Learning, Chapter 6.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Generating data from a mixture
clear all;close all;
path(path,'../utilities');
%% Define the mixture components
mixture_means = [3 3;1 -3];
mixture_covs(:,:,1) = [1 0;0 2];
mixture_covs(:,:,2) = [2 0;0 1];
priors = [0.7 0.3];

%% Generate data points one at a time
figure(1);hold off
plotpoints = [1:5 10:5:30 40 50];
X = [];
for n = 1:50
    %%
    % Flip a biased coin to choose from the prior
    comp = find(rand<cumsum(priors));
    comp = comp(1);
    X(n,:) = gausssamp(mixture_means(comp,:)',mixture_covs(:,:,comp),1);
    if any(plotpoints==n)
        figure(1);
        hold off
        plot(X(end,1),X(end,2),'ko','markersize',20,'markerfacecolor',[0.6 0.6 0.6]);
        hold on
        % Make the contours
        for k = 1:2
            plot_2D_gauss(mixture_means(k,:),mixture_covs(:,:,k),...
                -3:0.1:5,-6:0.1:6);
        end
        plot(X(:,1),X(:,2),'ko');
    end
end