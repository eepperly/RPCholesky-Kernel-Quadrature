close all
open('../figs/1d_errors.fig')
open('../figs/3d_errors.fig')
open('../figs/1d_times.fig')
open('../figs/3d_times.fig')

ds = [1 3];
s = 3;
ns = 2 .^ (2:7);
trials = 100;

prefactor = (-1)^(s-1)*(2*pi)^(2*s)/factorial(2*s);
bern = @(x) x.^6 - 3*x.^5 + 2.5*x.^4 - 0.5*x.^2 + 1/42;
k_1d = @(x,y) 1 + prefactor * bern(mod(x-y,1));
all_errors_copy = all_errors;
extra_colors = ["#4DBEEE","#77AC30"];
extra_markers = ["diamond","+"];
methods = ["pwkq","thinning"];

for d_idx = 1:length(ds)
    d = ds(d_idx);

    if d == 1
        k = k_1d;
        proposal = @() rand();
    elseif d == 2
        k = @(s,t) k_1d(s(1),t(1)) * k_1d(s(2),t(2));
        proposal = @() rand(2,1);
    elseif d == 3
        k = @(s,t) k_1d(s(1),t(1)) * k_1d(s(2),t(2)) * k_1d(s(3),t(3));
        proposal = @() rand(3,1);
    end

    for method_idx = 1:length(methods)
        method_name = methods{method_idx};

        errors = zeros(length(ns),trials);
        alltimes = zeros(length(ns),trials);
        for n_idx = 1:length(ns)
            n = ns(n_idx);
            load(sprintf("%s/%d_%d.mat",method_name,d,n))
            load(sprintf("%s/%d_%d_times.mat",method_name,d,n))
            for trial=1:trials
                S = reshape(pts(:,:,trial),[d n]);
                if strcmp(method_name, "pwkq")
                    w = weights(:,trial);
                else
                    w = ones(n,1) / n;
                end
                K = zeros(n,n);
                for i = 1:n
                    for j = i:n
                        K(i,j) = k(S(:,i),S(:,j));
                        K(j,i) = K(i,j);
                    end
                end
                errors(n_idx,trial) = sqrt(1 + w'*K*w-2*sum(w));

                alltimes(n_idx,trial) = times(trial);
            end
        end
    
        figure(d_idx)
        means = mean(errors,2);
        shadedErrorBar(ns,means,...
            [quantile(errors,0.9,2)-means means-quantile(errors,0.1,2)]',...
            'lineProps',{'-','Color',extra_colors{method_idx},...
            'Marker',extra_markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',extra_colors{method_idx}},...
            'patchSaturation',0.1); hold on
    
        errors_d = all_errors{d_idx};
        errors_d{end+1} = errors;
        all_errors_copy{d_idx} = errors_d;
        if d_idx == 1
            legend({'Monte Carlo','IID','CVS','RPCholesky (Ours)','PWKQ','Thinning'})
        end

        times = alltimes;
        meantimes = mean(times,2);
        figure(length(ds)+d_idx)
        shadedErrorBar(ns(1:length(ns)),meantimes,...
            [quantile(times,0.9,2)-meantimes meantimes-quantile(times,0.1,2)]',...
            'lineProps',{'-','Color',extra_colors{method_idx},...
            'Marker',extra_markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',extra_colors{method_idx}},...
            'patchSaturation',0.1); hold on
    end
end