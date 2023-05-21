function [S,L] = rpcholesky_opt(proposal,n,k)
%RPCHOLESKY_OPT Select points using randomly pivoted Cholesky with
%acceleration by optimization. Algorithm 4 in the main paper
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
s = proposal(); kmax = k(s,s);
warning('off','MATLAB:nearlySingularMatrix')
while i <= n
    trials = trials + 1;
    s = proposal();
    [d,c] = kernel_eval(L,i,S,s,k);
    if rand() < d / kmax
        S(:,i) = s;
        L(i,1:i) = [c' sqrt(d)];
        i = i+1;
        % fprintf('%d\t%d\n', i-1, trials)
        trials = 0;
    end
    if trials >= 25
        opts=optimoptions(@fminunc,'Display','off',...
            'OptimalityTolerance',1e-3);
        [~,kmax] = fminunc(@(x) -kernel_eval(L,i,S,x,k),proposal(),opts);
        kmax = -kmax;
        trials = 0;
    end
end
warning('on','MATLAB:nearlySingularMatrix')
end

function [d,c] = kernel_eval(L,i,S,s,k)
    kvals = zeros(i-1,1);
    for j = 1:length(kvals)
        kvals(j) = k(S(:,j),s);
    end
    c = L(1:(i-1),1:(i-1)) \ kvals;
    d = k(s,s) - norm(c)^2;
end