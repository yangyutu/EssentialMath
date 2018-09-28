function r1 = Vasicek3(M, N, r0)
% dr_t = k (rt - r_t) dt + sigma dW_t
 
k = 0.82;
dt = 1 / N;
theta = 0.05;
sigma = 0.10 / 365;
T = N / 365 ;
 
rep = r0 * ones(M + 1, N + 1);
z = sqrt(dt) * randn(M + 1, N + 1);
 
for i = 1 : N
rep(:, i + 1) = rep(:, i) + dt * k * (theta - rep(:, i)) + sigma * z(:, i) ;
end
 
Rf = zeros(1, M + 1);
 
for i = 1 : M + 1
Rf(i) = dt * sum(rep(i, :));
Rf(i) = exp(-Rf(i) * T);
end
 
r1 = mean(Rf);
 
end