%% approx_expected_value.m
% From A First Course in Machine Learning, Chapter 2.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Approximating expected values via sampling
clear all;close all;
%% We are trying to compute the expected value of
% $y^2$
%%
% Where
% $p(y)=U(0,1)$
%% 
% Which is given by:
% $\int y^2 p(y) dy$
%%
% The analytic result is:
% $\frac{1}{3}$
%% Generate samples
ys = rand(10000,1);
% compute the expectation
ey2 = mean(ys.^2);
fprintf('\nSample-based approximation: %g',ey2);
%% Look at the evolution of the approximation
posns = [1:10:length(ys)];
ey2_evol = zeros(size(posns));
for i = 1:length(posns)
    ey2_evol(i) = mean(ys(1:posns(i)).^2);
end
figure(1);hold off
plot(posns,ey2_evol);
hold on
plot([posns(1) posns(end)],[1/3 1/3],'r--');
xlabel('Samples');
ylabel('Approximation');
