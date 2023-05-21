% Benchmark test. Produces figure 2

addpath('..')

trials = 100;
s = 3;
prefactor = (-1)^(s-1)*(2*pi)^(2*s)/factorial(2*s);
bern = @(x) x.^6 - 3*x.^5 + 2.5*x.^4 - 0.5*x.^2 + 1/42;
k_1d = @(x,y) 1 + prefactor * bern(mod(x-y,1));
ns = 2.^(2:10);
num_err_plt = 6;

methods = {{@iid_sample, @uniform_weights, true, false, 7, 'Monte Carlo'},...
           {@iid_sample, @optimal_weights, true, false, 7, 'IID'},...
           {@dpp, @optimal_weights, true, true, 8, 'CVS'},...
           {@rpcholesky_opt, @optimal_weights, true, true, 9,...
                'RPCholesky (Ours)'}};
colors = ["#EDB120","#D95319","#7E2F8E","#0072BD"];
markers = ["*","o","^","s"];

ds = [1 3];

close all

all_errors = cell(length(ds),1);
all_times  = cell(length(ds),1);

for d_idx = 1:length(ds)
    d = ds(d_idx);
    errors_d = cell(length(methods),1);
    times_d  = cell(length(methods),1);

    if d == 1
        k = k_1d;
        proposal = @() rand();
    elseif d == 3
        k = @(s,t) k_1d(s(1),t(1)) * k_1d(s(2),t(2)) * k_1d(s(3),t(3));
        proposal = @() rand(3,1);
    end

    method_names = {};
    for method_idx = 1:length(methods)
        method_items = methods{method_idx};
        nodes_method = method_items{1};
        weights_method = method_items{2};
        is_random = method_items{3};
        plt_times = method_items{4};
        max_idx_n = method_items{5};
        name = method_items{6};
        if is_random
            errors = zeros(max_idx_n,trials);
            times  = zeros(max_idx_n,trials);
        else
            errors = zeros(max_idx_n, 1);
            times  = zeros(max_idx_n, 1);
        end
        for idx_n = 1:max_idx_n
            n = ns(idx_n);
            tic
            if is_random
                for trial = 1:trials
                    tic
                    S = nodes_method(proposal,n,k);
                    times(idx_n,trial) = toc;
                    w = weights_method(k,S,@(s) 1);
                    K = kernel_matrix(k,S);
                    errors(idx_n,trial) = sqrt(1 + w'*K*w-2*sum(w));
                    fprintf('%s\t%d/%d\t%e\n', name, n, trial,...
                        mean(errors(idx_n,1:trial)));
                end
            else
                tic
                S = nodes_method(proposal,n,k);
                times(idx_n,trial) = toc;
                w = weights_method(k,S,@(s) 1);
                K = kernel_matrix(k,S);
                errors(idx_n) = sqrt(1 + w'*K*w-2*sum(w));
                fprintf('%s\t%d/%d\t%e\n', name, n, trial, errors(idx_n));
            end
        end
        errors_d{method_idx} = errors;
        times_d{method_idx} = errors;

        errors = errors(1:num_err_plt,1:trials);
        means = mean(errors,2);
        figure(d_idx)
        shadedErrorBar(ns(1:num_err_plt),means,...
            [quantile(errors,0.9,2)-means means-quantile(errors,0.1,2)]',...
            'lineProps',{'-','Color',colors{method_idx},...
            'Marker',markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',colors{method_idx}},...
            'patchSaturation',0.1); hold on
        set(gca,'XScale','log')
        set(gca,'YScale','log')
        xlabel('Number of nodes $n$')
        ylabel('$\mathrm{Err}(\mathsf{S},\mbox{\boldmath $w$};g)$')
        drawnow

        if plt_times
            meantimes = mean(times,2);
            figure(length(ds)+d_idx)
            shadedErrorBar(ns(1:max_idx_n),meantimes,...
                [quantile(times,0.9,2)-meantimes meantimes-quantile(times,0.1,2)]',...
                'lineProps',{'-','Color',colors{method_idx},...
                'Marker',markers{method_idx},...
                'MarkerSize',12,'MarkerFaceColor',colors{method_idx}},...
                'patchSaturation',0.1); hold on
            set(gca,'XScale','log')
            set(gca,'YScale','log')
            xlabel('Number of nodes $n$')
            ylabel('Time (sec)')
            drawnow
        end

        method_names{end+1} = name;
    end
    all_errors{d_idx} = errors_d;
    if d == ds(1)
        figure(d_idx)
        legend(method_names,"Location","Southwest")
    end

    saveas(gcf,sprintf('../figs/%dd.fig',d))
    saveas(gcf,sprintf('../figs/%dd.png',d))
end