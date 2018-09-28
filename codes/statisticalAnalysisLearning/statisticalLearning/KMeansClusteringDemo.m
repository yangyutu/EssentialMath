clear all
close all

rng(1); % For reproducibility
X = [randn(100,2)*0.75+ones(100,2);
    randn(100,2)*0.5-ones(100,2);
    randn(100,2)*0.25 - 2*ones(100,2)];

figure(1);
plot(X(:,1),X(:,2),'.');
title 'Randomly Generated Data';

opts = statset('Display','final');
[idx,C] = kmeans(X,3,'Distance','sqeuclidean',...
    'Replicates',5,'Options',opts);

figure(2);
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)

plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%% nonlinear data set
theta = randn(200,1)*0.3;
X = [10*cos(theta(1:100)) 10*sin(theta(1:100));
    8*cos(theta(1:100)) 8*sin(theta(1:100))];

figure(3);
plot(X(:,1),X(:,2),'.');
title 'Randomly Generated Data';

opts = statset('Display','final');
[idx,C] = kmeans(X,2,'Distance','sqeuclidean',...
    'Replicates',5,'Options',opts);

figure(4);
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)


plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

