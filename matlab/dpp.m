function S = dpp(proposal,n,k)
%DPP Select points using continuous volume sampling using
%Metropolis-Hastings MCMC sampler
%   Inputs:
%   - proposal: a function which returns a random sample from k(x,x)
%   dmu(x), stored as d*1 column vectors
%   - n: number of points to select
%   - k: kernel function
%   Outputs:
%   - S: a d*n matrix of the selected points

S = iid_sample(proposal,n,k);

K = kernel_matrix(k,S);
detK = det(K);

for trial = 1:10*n
    i = randsample(n,1);
    s = proposal();

    Kprop = K;
    for j = 1:n
        if i == j
            Kprop(i,i) = k(s,s);
        else
            Kprop(i,j) = k(S(:,j),s);
            Kprop(j,i) = Kprop(i,j);
        end
    end
    detKprop = det(Kprop);

    if rand() < 0.5*min(1,detKprop/detK)
        S(:,i) = s;
        K = Kprop;
        detK = detKprop;
    end
end
end