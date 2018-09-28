clear all
close all

%Generate 100 points uniformly distributed in the unit disk. 
rng(1); % For reproducibility
r = sqrt(rand(100,1)*0.5); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points

%Generate 100 points uniformly distributed in the annulus.
r2 = sqrt(3*rand(100,1)*0.3+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points


figure(1)
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15)
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15)
%ezpolar(@(x)1);ezpolar(@(x)2);
axis equal
hold off

data3 = [data1;data2];

for i = 1 : size(data3,1)
    map(i,1) = data3(i,1);
    map(i,2) = data3(i,2);
    map(i,3) = data3(i,1)^2 + data3(i,2)^2;
end
figure(2)
plot3(map(1:100,1),map(1:100,2),map(1:100,3),'r.','MarkerSize',15);
hold on
plot3(map(101:end,1),map(101:end,2),map(101:end,3),'b.','MarkerSize',15);