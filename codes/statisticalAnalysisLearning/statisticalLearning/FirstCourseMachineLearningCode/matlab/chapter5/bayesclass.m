%% bayesclass.m
% From A First Course in Machine Learning, Chapter 5.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Bayesian classifier
clear all;close all;

%% Load the data
load ../data/bc_data

% Plot the data

cl = unique(t);
col = {'ko','kd','ks'}
fcol = {[1 0 0],[0 1 0],[0 0 1]};
figure(1);
hold off
for c = 1:length(cl)
    pos = find(t==cl(c));
    plot(X(pos,1),X(pos,2),col{c},...
        'markersize',10,'linewidth',2,...
        'markerfacecolor',fcol{c});
    hold on
end
xlim([-3 7])
ylim([-6 6])



%% Fit class-conditional Gaussians for each class
% Using the Naive (independence) assumption
for c = 1:length(cl)
    pos = find(t==cl(c));
    % Find the means
    class_mean(c,:) = mean(X(pos,:));
    class_var(c,:) = var(X(pos,:),1);
end


%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-3:0.1:7,-6:0.1:6);
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    tempc = diag(class_var(c,:));
    const = -log(2*pi) - log(det(tempc));
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv));;
end

Probs = Probs./repmat(sum(Probs,3),[1,1,3]);

%% Plot the predictive contours
figure(1);hold off
for i = 1:3
    subplot(1,3,i);
    hold off
    for c = 1:length(cl)
        pos = find(t==cl(c));
        plot(X(pos,1),X(pos,2),col{c},...
            'markersize',10,'linewidth',2,...
            'markerfacecolor',fcol{c});
        hold on
    end
    xlim([-3 7])
    ylim([-6 6])
    
    contour(Xv,Yv,Probs(:,:,i));
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end


%% Repeat without Naive assumption
class_var = [];
for c = 1:length(cl)
    pos = find(t==cl(c));
    % Find the means
    class_mean(c,:) = mean(X(pos,:));
    class_var(:,:,c) = cov(X(pos,:),1);
end


%% Compute the predictive probabilities
[Xv,Yv] = meshgrid(-3:0.1:7,-6:0.1:6);
Probs = [];
for c = 1:length(cl)
    temp = [Xv(:)-class_mean(c,1) Yv(:)-class_mean(c,2)];
    tempc = class_var(:,:,c);
    const = -log(2*pi) - log(det(tempc));
    Probs(:,:,c) = reshape(exp(const - 0.5*diag(temp*inv(tempc)*temp')),size(Xv));;
end

Probs = Probs./repmat(sum(Probs,3),[1,1,3]);

%% Plot the predictive contours
figure(1);hold off
for i = 1:3
    subplot(1,3,i);
    hold off
    for c = 1:length(cl)
        pos = find(t==cl(c));
        plot(X(pos,1),X(pos,2),col{c},...
            'markersize',10,'linewidth',2,...
            'markerfacecolor',fcol{c});
        hold on
    end
    xlim([-3 7])
    ylim([-6 6])
    
    contour(Xv,Yv,Probs(:,:,i));
    ti = sprintf('Probability contours for class %g',i);
    title(ti);
end