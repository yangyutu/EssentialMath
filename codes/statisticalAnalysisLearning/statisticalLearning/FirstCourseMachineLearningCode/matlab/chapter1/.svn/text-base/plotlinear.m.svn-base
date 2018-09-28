%% Plotlinear.m
% From A First Course in Machine Learning, Chapter 1.
% Simon Rogers, 31/10/11 [simon.rogers@glasgow.ac.uk]
clear all;close all;

%% Define two points for the x-axis
x = [-5 5];

%% Define the different intercepts and gradients to plot
w0 = [0:1:20];
w1 = [0:0.4:8];

%% Plot all of the lines

figure(1);
hold off

for i = 1:length(w0)
    plot(x,w0(i)+w1(i).*x);
    hold all
    fprintf('\ny = %g + %g x',w0(i),w1(i));
end

%% Request user input
close all;
figure(1);hold off
fprintf('\nKeeps plotting lines on the current plot until you quit (ctrl-c)\n');
while 1
    intercept = str2num(input('Enter intercept:','s'));
    gradient = str2num(input('Enter gradient:','s'));
    plot(x,intercept + gradient.*x);hold all
    fprintf('\n y = %g + %g x\n\n',intercept,gradient);
end