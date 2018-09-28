%% svmroc.m
% From A First Course in Machine Learning, Chapter 5.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% ROC analysis of SVM
clear all;close all;
%% Load the data
load ../data/SVMdata2
load ../data/SVMtest

%% Compute the kernels
gam = 10; % Experiment with this value
N = size(X,1);
Nt = size(testX,1);
for n = 1:N
    for n2 = 1:N
        K(n,n2) = exp(-gam*sum((X(n,:)-X(n2,:)).^2));
    end
    for n2 = 1:Nt
        testK(n,n2) = exp(-gam*sum((X(n,:)-testX(n2,:)).^2));
    end
end

%% Train the SVM
H = (t*t').*K + 1e-5*eye(N);
f = repmat(1,N,1);
A = [];b = [];
LB = repmat(0,N,1); UB = repmat(inf,N,1);
Aeq = t';beq = 0;

% Fix C
C = 10;
UB = repmat(C,N,1);
% Following line runs the SVM
alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB);

fout = sum(repmat(alpha.*t,1,N).*K,1)';
pos = find(alpha>1e-6);
bias = mean(t(pos)-fout(pos));

%% Compute the test predictions
testpred = (alpha.*t)'*testK + bias;
testpred = testpred';

%% Do the ROC analysis

th_vals = [min(testpred):0.01:max(testpred)+0.01];
sens = [];
spec = [];
for i = 1:length(th_vals)
    b_pred = testpred>=th_vals(i);
    % Compute true positives, false positives, true negatives, true
    % positives
    TP = sum(b_pred==1 & testt == 1);
    FP = sum(b_pred==1 & testt == -1);
    TN = sum(b_pred==0 & testt == -1);
    FN = sum(b_pred==0 & testt == 1);
    % Compute sensitivity and specificity
    sens(i) = TP/(TP+FN);
    spec(i) = TN/(TN+FP);
end

%% Plot the ROC curve
figure(1);hold off
cspec = 1-spec;
cspec = cspec(end:-1:1);
sens = sens(end:-1:1);
plot(cspec,sens,'k')

%% Compute the AUC
AUC = sum(0.5*(sens(2:end)+sens(1:end-1)).*(cspec(2:end) - cspec(1:end-1)));
fprintf('\nAUC: %g',AUC);

