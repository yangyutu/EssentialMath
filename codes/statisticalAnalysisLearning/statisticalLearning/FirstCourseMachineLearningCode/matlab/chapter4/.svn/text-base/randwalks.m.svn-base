%% randwalks.m
% From A First Course in Machine Learning, Chapter 4.
% Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
% Random walk examples
clear all;close all;

%% Define the starting point
w = [0;0];

%% Define the jump covariance
si = [0.1 0;0 0.1];

%% Do N steps
path(path,'../utilities');
N = 10;
figure(1);
hold off
plot(w(1),w(2),'ko','markersize',10);
hold on
for n = 1:N
    wnew = gausssamp(w,si,1)';
    plot(wnew(1),wnew(2),'ko','markersize',10);
    plot([w(1) wnew(1)],[w(2) wnew(2)],'k');
    w = wnew;
end
xlim([-10 10]);
ylim([-10 10]);


%% Second example
w = [-2;-2];
si = [1 0;0 5];
N = 10;
figure(1);
plot(w(1),w(2),'ro','markersize',10);
hold on
for n = 1:N
    wnew = gausssamp(w,si,1)';
    plot(wnew(1),wnew(2),'ro','markersize',10);
    plot([w(1) wnew(1)],[w(2) wnew(2)],'r');
    w = wnew;
end
xlim([-10 10]);
ylim([-10 10]);
