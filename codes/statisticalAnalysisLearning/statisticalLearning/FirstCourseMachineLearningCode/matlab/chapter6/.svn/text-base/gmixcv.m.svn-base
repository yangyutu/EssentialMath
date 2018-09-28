%% gmixcv.m
% From A First Course in Machine Learning, Chapter 6.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% CV for selection of K in a Gaussian Mixture
clear all;close all;
path(path,'../utilities');

%% Load the data
load ../data/kmeansdata


%% Plot the data
figure(1);hold off
plot(X(:,1),X(:,2),'ko');


%% Do a 10-fold CV
NFold = 10;
N = size(X,1);
sizes = repmat(floor(N/NFold),1,NFold);
sizes(end) = sizes(end) + N - sum(sizes);
csizes = [0 cumsum(sizes)];
order = randperm(N); % Randomise the data order
MaxIts = 100;
%% Loop over K
Kvals = 1:10;
for kv = 1:length(Kvals)
    K = Kvals(kv);
    for fold = 1:NFold
        fprintf('\nK: %g, fold: %g',K,fold);
        foldindex = order(csizes(fold)+1:csizes(fold+1));
        trainX = X;
        trainX(foldindex,:) = [];
        testX = X(foldindex,:);
        
        % Train the mixture
        means = randn(K,2);
        for k = 1:K
            covs(:,:,k) = rand*eye(2);
        end
        priors = repmat(1/K,1,K);
        B = [];
        B(1) = -inf;
        converged = 0;
        it = 0;
        tol = 1e-2;
        Ntrain = size(trainX,1);
        D = size(X,2);
        while 1
            it = it + 1;
            % Update q
            temp = zeros(Ntrain,K);
            for k = 1:K
                const = -(D/2)*log(2*pi) - 0.5*log(det(covs(:,:,k)));
                Xm = trainX - repmat(means(k,:),Ntrain,1);
                temp(:,k) = const - 0.5 * diag(Xm*inv(covs(:,:,k))*Xm');
            end

            % Compute the Bound on the likelihood
            if it>1
                B(it) = sum(sum(q.*log(repmat(priors,Ntrain,1)))) + ...
                    sum(sum(q.*temp)) - ...
                    sum(sum(q.*log(q)));
                if abs(B(it)-B(it-1))<tol
                    converged = 1;

                end
            end

            if converged == 1 || it>MaxIts
                break
            end

            temp = temp + repmat(priors,Ntrain,1);

            q = exp(temp - repmat(max(temp,[],2),1,K));
            % Minor hack for numerical issues - stops the code crashing when
            % clusters are empty - would be better to use MAP.
            q(q<1e-3) = 1e-3;
            q(q>1-1e-3) = 1-1e-3;
            q = q./repmat(sum(q,2),1,K);
            % Update priors
            priors = mean(q,1);
            % Update means
            for k = 1:K
                means(k,:) = sum(trainX.*repmat(q(:,k),1,D),1)./sum(q(:,k));
            end
            % update covariances
            for k = 1:K
                Xm = trainX - repmat(means(k,:),Ntrain,1);
                covs(:,:,k) = (Xm.*repmat(q(:,k),1,D))'*Xm;
                covs(:,:,k) = covs(:,:,k)./sum(q(:,k));
            end


        end
        
        
        
        % Compute the held-out likelihood
        Ntest = size(testX,1);
        temp = zeros(Ntest,K);
        for k = 1:K
            const = -(D/2)*log(2*pi) - 0.5*log(det(covs(:,:,k)));
                Xm = testX - repmat(means(k,:),Ntest,1);
                temp(:,k) = const - 0.5 * diag(Xm*inv(covs(:,:,k))*Xm');
                
        end
        temp = exp(temp).*repmat(priors,Ntest,1);
        outlike(kv,csizes(fold)+1:csizes(fold+1)) = log(sum(temp,2))';
        
        
    end
end

%% Plot the evolution in log likelihood
errorbar(Kvals,mean(outlike,2),std(outlike,1,2)./sqrt(N));
xlabel('K');
ylabel('Average held-out likelidood (+- standard error)');