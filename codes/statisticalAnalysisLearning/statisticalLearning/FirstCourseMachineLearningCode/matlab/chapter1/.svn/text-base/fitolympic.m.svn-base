%% fitolympic.m
% From A First Course in Machine Learning, Chapter 1.
% Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]
clear all;close all;

%% Load the Olympic data
load ../data/olympics

%% Extract the male 100m data
x = male100(:,1); % Olympic years
t = male100(:,2); % Winning times

% Change the preceeding lines for different data.  e.g.
% x = male400(:,1); % Olympic years
% t = male400(:,2); % Winning times
% for the mens 400m event.

N = length(x); % 27
%% Compute the various averages required
% $\frac{1}{N}\sum_n x_n$
m_x = sum(x)/N;
%%
% $$\frac{1}{N}\sum_n t_n$$
%
m_t = sum(t)/N;
%%
% $\frac{1}{N}\sum_n t_n x_n$
m_xt = sum(t.*x)/N;
%%
% $\frac{1}{N}\sum_n x_n^2$
m_xx = sum(x.*x)/N;


%% Compute w1 (gradient) (Equation 1.10)
w_1 = (m_xt - m_x*m_t)/(m_xx - m_x^2);
%% Compute w0 (intercept) (Equation 1.8)
w_0 = m_t - w_1*m_x;

%% Plot the data and linear fit
figure(1);hold off
plot(x,t,'bo','markersize',10,'linewidth',2)
xplot = [min(x)-4 max(x)+4];
xlim(xplot);
hold on
plot(xplot,w_0+w_1*xplot,'r','linewidth',2)
xlabel('Olympic year');
ylabel('Winning time');