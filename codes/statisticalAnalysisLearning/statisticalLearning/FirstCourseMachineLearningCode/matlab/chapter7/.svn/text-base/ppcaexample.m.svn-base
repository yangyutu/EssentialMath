%% ppcaexample.m
% From A First Course in Machine Learning, Chapter 6.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Probabilistic PCA example using Variational Bayes
clear all;close all;

%% Generate the data
Y = [randn(20,2);randn(20,2)+5;randn(20,2)+repmat([5 0],20,1);randn(20,2)+repmat([0 5],20,1)];
% Add 5 random dimensions
N = size(Y,1);
Y = [Y randn(N,5)];
% labels - just used for plotting
t = [repmat(1,20,1);repmat(2,20,1);repmat(3,20,1);repmat(4,20,1)];
%% Plot the original data
symbs = {'ro','gs','b^','kd'};
figure(1);hold off
for k = 1:4
    pos = find(t==k);
    plot(Y(pos,1),Y(pos,2),symbs{k});
    hold on
end


%% Initialise the parameters
a = 1;b = 1; % Hyper-parameters for precision
e_tau = a/b;
[N,M] = size(Y);
D = 2; % Number of projection dimensions
e_w = randn(M,D);
for m = 1:M
    e_wwt(:,:,m) = eye(D) + e_w(m,:)'*e_w(m,:);
end
tol = 1e-3;

%% Run the algorithm

MaxIts = 100;
for it = 1:MaxIts
    % Update X
    % Compute \Sigma_x - this is the same for all x
    sigx = inv(eye(D) + e_tau*sum(e_wwt,3));
    for n = 1:N
        e_X(n,:) = e_tau*sigx*sum(e_w.*repmat(Y(n,:),D,1)',1)';
        e_XXt(:,:,n) = sigx + e_X(n,:)'*e_X(n,:);
    end
    
    % Update W
    sigw = inv(eye(D) + e_tau*sum(e_XXt,3));
    for m = 1:M
        e_w(m,:) = e_tau*sigw*sum(e_X.*repmat(Y(:,m),1,D),1)';
        e_wwt(:,:,m) = sigw + e_w(m,:)'*e_w(m,:);
    end
    
    % Update tau
    e = a + N*M/2;
    % Compute the nasty outer expectation.  Note that these two loops could
    % be made *much* more efficient
    outer_expect = 0;
    for n = 1:N
        for m = 1:M
            outer_expect = outer_expect + ...
                trace(e_wwt(:,:,m)*sigx) + ...
                e_X(n,:)*e_wwt(:,:,m)*e_X(n,:)';
        end
    end
    f = b + 0.5*sum(sum(Y.^2)) - sum(sum(e_X*e_w')) + ...
        0.5*outer_expect;
    e_tau = e/f;
    e_log_tau = mean(log(gamrnd(e,1/f,[1000,1])));
    
    
    
    % Compute the bound
    LB = a*log(b) + (a-1)*e_log_tau - b*e_tau - gammaln(a);
    LB = LB - (e*log(f) + (e-1)*e_log_tau - f*e_tau - gammaln(e));
    

    for n = 1:N
        LB = LB + (...
            -(D/2)*log(2*pi) - 0.5*(trace(sigx) + e_X(n,:)*e_X(n,:)'));
        LB = LB - (...
            -(D/2)*log(2*pi) - 0.5*log(det(sigx)) - 0.5*D);
    end
    
    for m = 1:M
        LB = LB + (...
            - (D/2)*log(2*pi) - 0.5*(trace(sigw) + e_w(m,:)*e_X(m,:)'));
        LB = LB - (...
            -(D/2)*log(2*pi) - 0.5*log(det(sigw)) - 0.5*D);
    end
    
    outer_expect = 0;
    for n = 1:N
        for m = 1:M
            outer_expect = outer_expect + trace(e_wwt(:,:,m)*sigx) + e_X(n,:)*e_wwt(:,:,m)*e_X(n,:)';
        end
    end
    
    % likelihood bit
    LB = LB + (...
        -(N*M/2)*log(2*pi) + (N*M/2)*e_log_tau - ...
        0.5*e_tau*(sum(sum(Y.^2)) - 2*sum(sum(Y.*(e_w*e_X')')) + outer_expect));
    B(it) = LB;
    
    if it>2
        if abs(B(it)-B(it-1))<tol
            break
        end
    end
end

%% Plot the bound
figure(1);hold off
plot(B);
xlabel('Iterations');
ylabel('Bound');

%% Plot the projection
figure(1);hold off
for k = 1:4
    pos = find(t==k);
    plot(e_X(pos,1),e_X(pos,2),symbs{k});
    hold on
end
title('Projection');
