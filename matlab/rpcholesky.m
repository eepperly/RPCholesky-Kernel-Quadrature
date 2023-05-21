function [S,L] = rpcholesky(proposal,n,k)
%RPCHOLESKY Select points using randomly pivoted Cholesky. Algorithm 2 in
%the main paper
%   Inputs:
%   - proposal: a function which returns a random sample from k(x,x)
%   dmu(x), stored as d*1 column vectors
%   - n: number of points to select
%   - k: kernel function
%   Outputs:
%   - S: a d*n matrix of the selected points
%   - L: a lower triangular Cholesky factor of the kernel matrix k(S,S)

S = zeros(length(proposal()),n);
i = 1;
L = zeros(n);
trials = 0;
while i <= n
    trials = trials + 1;
    s = proposal();
    kvals = zeros(i-1,1);
    for j = 1:length(kvals)
        kvals(j) = k(S(:,j),s);
    end
    c = L(1:(i-1),1:(i-1)) \ kvals;
    kss = k(s,s);
    d = kss - norm(c)^2;
    if rand() < d / kss
        S(:,i) = s;
        L(i,1:i) = [c' sqrt(d)];
        i = i+1;
        trials = 0;
    end
end
end