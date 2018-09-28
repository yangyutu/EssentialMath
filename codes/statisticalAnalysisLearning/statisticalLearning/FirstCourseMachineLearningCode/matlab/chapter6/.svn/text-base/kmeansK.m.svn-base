%% kmeansK.m
% From A First Course in Machine Learning, Chapter 6.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Overfitting in K-means
clear all;close all;

%% Load the data
load ../data/kmeansdata

%% Plot the data
figure(1);hold off
plot(X(:,1),X(:,2),'ko');

Kvals = 1:10;
Nreps = 50;
total_distance = zeros(length(Kvals),Nreps);
for kv = 1:length(Kvals)
    for rep = 1:Nreps
        K = Kvals(kv);
        cluster_means = rand(K,2)*10-5;
        converged = 0;
        N = size(X,1);
        cluster_assignments = zeros(N,K);
        di = zeros(N,K);
        while ~converged
   
            % Update assignments
            for k = 1:K
                di(:,k) = sum((X - repmat(cluster_means(k,:),N,1)).^2,2);
            end
            old_assignments = cluster_assignments;
            cluster_assignments = (di == repmat(min(di,[],2),1,K));
            if sum(sum(old_assignments~=cluster_assignments))==0
                converged = 1;
            end

            % Update means
            for k = 1:K   
                if sum(cluster_assignments(:,k))==0
                    % This cluster is empty, randomise it
                    cluster_means(k,:) = rand(1,2)*10-5;
                else
                    cluster_means(k,:) = mean(X(cluster_assignments(:,k),:),1);
                end
            end
        end
        % Compute the distance
        total_distance(kv,rep) = sum(sum(di.*cluster_assignments));
        
    end
end

%% Make the boxplot
figure(1);hold off
boxplot(log(total_distance)');
xlabel('K');
ylabel('Log D');
