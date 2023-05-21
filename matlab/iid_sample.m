function S = iid_sample(proposal,n,varargin)
%IID_SAMPLE Return points sampled iid from a proposal
%   Inputs:
%   - proposal: a function which returns a random sample from k(x,x)
%   dmu(x), stored as d*1 column vectors
%   - n: number of points to select
%   - k: kernel function
%   Outputs:
%   - S: a d*n matrix of the selected points

S = zeros(length(proposal()),n);
for i = 1:n
    S(:,i) = proposal();
end
end