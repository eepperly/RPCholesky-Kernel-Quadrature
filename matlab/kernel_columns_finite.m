function K = kernel_columns_finite(data,k,S)
K = k(data,data(S,:));
end