% QM9 chemistry test. Produces figure 3

addpath('..')

if exist('qm9.mat','file') == 2
    load('qm9.mat')
else
    error(['qm9.mat is not present. To generate, run ' ...
        '../../python/qm9/setup.sh'])
end

N = 2e4;
X = X(1:N, :); Y = Y(1:N,5);
target = mean(Y);

bandwidth = 0.3955; 
k = @(X1,X2) exp(-pdist2(X1,X2).^2/(2*bandwidth^2));

trials = 100;
ns = 2.^(2:10);
num_err_plt = 8;
plt_times = true;

methods = { {@greedy_finite, @optimal_weights_finite, 8, true, 'SBQ'},...
            {@(XX,nn,kk) randsample(N,nn),...
                @(XX,kk,SS) ones(length(SS),1)/length(SS), 8, false, 'Monte Carlo'},...
            {@(XX,nn,kk) randsample(N,nn), @optimal_weights_finite, 8, false, 'IID'},...
            {@rpcholesky_finite, @optimal_weights_finite, 8, false, 'RPCholesky (Ours)'} };
colors = ["#000000","#EDB120","#D95319","#0072BD"];
markers = ["v","*","o","s"];

close all

all_worst = cell(length(methods),1);
all_polarizability  = cell(length(methods),1);

method_names = {};
for method_idx = 1:length(methods)
    method_items = methods{method_idx};
    nodes_method = method_items{1};
    weights_method = method_items{2};
    max_idx_n = method_items{3};
    deterministic = method_items{4};
    name = method_items{5};
    if deterministic
        current_trials = 1;
    else
        current_trials = trials;
    end
    worst = zeros(max_idx_n,current_trials);
    polarizability  = zeros(max_idx_n,current_trials);
    for idx_n = 1:max_idx_n
        n = ns(idx_n);
        for trial = 1:current_trials
            S = nodes_method(X,n,k);
            w = weights_method(X,k,S);
            K = k(X(S,:),X(S,:));
            cols = kernel_columns_finite(X,k,S);
            worst(idx_n,trial) = sqrt(total_mean+w'*K*w-2*mean(cols*w));
            polarizability(idx_n,trial) = abs(target - w'*Y(S)) / abs(target);
            fprintf('%s\t%d/%d\t%e\t%e\n', name, n, trial,...
                mean(worst(idx_n,1:trial)),mean(polarizability(idx_n,1:trial)));
        end
    end
    all_worst{method_idx} = worst;
    all_polarizability{method_idx} = polarizability;

    worst = worst(1:num_err_plt,1:current_trials);
    means = mean(worst,2);
    figure(1)
    if deterministic
        plot(ns(1:num_err_plt),means,'-','Color',colors{method_idx},...
            'Marker',markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',colors{method_idx}); hold on
    else
        shadedErrorBar(ns(1:num_err_plt),means,...
            [quantile(worst,0.9,2)-means means-quantile(worst,0.1,2)]',...
            'lineProps',{'-','Color',colors{method_idx},...
            'Marker',markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',colors{method_idx}},...
            'patchSaturation',0.1); hold on
    end
    set(gca,'XScale','log')
    set(gca,'YScale','log')
    xlabel('Number of nodes $n$')
    ylabel('$\mathrm{Err}(\mathsf{S},\mbox{\boldmath $w$};g)$')
    drawnow

    polarizability = polarizability(1:num_err_plt,1:current_trials);
    means = mean(polarizability,2);
    figure(2)
    if deterministic
        plot(ns(1:num_err_plt),means,'-','Color',colors{method_idx},...
            'Marker',markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',colors{method_idx}); hold on
    else
        shadedErrorBar(ns(1:num_err_plt),means,...
            [quantile(polarizability,0.9,2)-means means-quantile(polarizability,0.1,2)]',...
            'lineProps',{'-','Color',colors{method_idx},...
            'Marker',markers{method_idx},...
            'MarkerSize',12,'MarkerFaceColor',colors{method_idx}},...
            'patchSaturation',0.1); hold on
    end
    set(gca,'XScale','log')
    set(gca,'YScale','log')
    xlabel('Number of nodes $n$')
    ylabel('Isotropic polarizability error')
    drawnow

    method_names{end+1} = name;
end
figure(1)
legend(method_names,"Location","Southwest")

figure(1)
saveas(gcf,'../figs/chemistry_worst.fig')
saveas(gcf,'../figs/chemistry_worst.png')

figure(2)
saveas(gcf,'../figs/chemistry_polarizability.fig')
saveas(gcf,'../figs/chemistry_polarizability.png')

save('chemistry.mat')