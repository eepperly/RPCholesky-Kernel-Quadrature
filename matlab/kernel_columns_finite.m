function K = kernel_columns_finite(data,k,S)
%KERNEL_COLUMNS_FINITE Compute selected columns of kernel matrix
%   Inputs:
%   - data: data stored as a |X| * d array
%   - k: kernel function
%   - S: a 1*n matrix of the indices of the selected points
%   Outputs:
%   - K: a |X| * n matrix storing the columns of the kernel matrix indexed
%        by the data points in S
K = k(data,data(S,:));
end