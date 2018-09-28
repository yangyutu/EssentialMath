function g = gausssamp(mu,sigma,N,sigmachol)

if ~exist('sigmachol')
    [sigmachol,p] = chol(sigma);
    sigmachol = sigmachol';
end
q = randn(length(mu),N);
g = repmat(mu,1,N)+ sigmachol*q;
g = g';