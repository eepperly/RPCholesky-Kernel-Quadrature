% Test of RPCholesky kernel quadrature on moon-shaped region
addpath('..')
close all
rho = 2;
k = @(x,y) (1+sqrt(5)*norm(x-y)/rho+5*norm(x-y)^2/3/rho^2)...
    *exp(-sqrt(5)*norm(x-y)/rho);

%% Integrators
integrate_circle = @(f,x,y,r) sum(sum(chebfun2(@(s,t) ...
    s*f(x+s*cos(t),y+s*sin(t)),[0 r -pi pi],'vectorize')));
integrate_shape = @(f) integrate_circle(@(x,y) (x^2+y^2)*f(x,y),0,0,2)...
    - integrate_circle(@(x,y) (x^2+y^2)*f(x,y),1,0,1);
integrator = @(s) integrate_shape(@(x,y) k(s,[x;y]));

%% Make contour plot
[S,L] = rpcholesky(@propose_special_region,20,k);

figure
nplt = 1000;
xx = linspace(-2,2,nplt);
yy = linspace(-2,2,nplt);
[xx,yy] = meshgrid(xx,yy);
zz = zeros(size(xx));
for i = 1:nplt
    for j = 1:nplt
        x = xx(i,j);
        y = yy(i,j);
        if x^2 + y^2 > 4 || (x-1)^2 + y^2 <= 1
            zz(i,j) = NaN;
            continue
        end
        kvals = zeros(length(S),1);
        for t = 1:length(kvals)
            kvals(t) = k(S(:,t),[x;y]);
        end
        zz(i,j) = k([x;y],[x;y]) - norm(L\kvals)^2;
    end
end
contourf(xx,yy,zz); hold on
scatter(S(1,:),S(2,:),'filled','k','SizeData',100)
colormap('cool')
axis square

%% Error plot
methods = { {@iid_sample,@uniform_weights,'Monte Carlo'},...
            {@iid_sample,@optimal_weights,'IID KQ'},...
            {@rpcholesky_opt,@optimal_weights, 'RPCholesky (Ours)'} };
all_errors = cell(size(methods));

trials = 100;
ns = 10:10:100;
f = @(x,y) sin(x)*exp(y);

for method_idx = 1:length(methods)
    errors = zeros(length(ns),trials);
    my_method = methods{method_idx};
    get_nodes = my_method{1};
    get_weights = my_method{2};
    name = my_method{3};
    for idx1 = 1:length(ns)
        n = ns(idx1);
        for trial = 1:trials
            S = get_nodes(@propose_special_region,n,k);
            w = get_weights(k,S,integrator,integrate_shape(@(x,y) 1));
            intf = integrate_shape(f);
            inta = 0;
            for i = 1:n
                inta = inta + w(i) * f(S(1,i),S(2,i));
            end
            errors(idx1,trial) = abs(intf - inta) / abs(intf);
            fprintf('%s\t%d\t%d\t%e\n',name,n,trial,mean(errors(idx1,1:trial)))
        end
    end
    all_errors{method_idx} = errors;
end

%% Plot
colors = ["#EDB120","#D95319","#0072BD"];
markers = ["*","o","s"];

figure
for i = 1:length(all_errors)
    errors = all_errors{i};
    means = mean(errors,2);
    shadedErrorBar(ns,means,...
        [quantile(errors,0.9,2)-means means-quantile(errors,0.1,2)]',...
        'lineProps',{'-','Color',colors{i},'Marker',markers{i},...
        'MarkerSize',12,'MarkerFaceColor',colors{i}},...
        'patchSaturation',0.1); hold on
end
set(gca,'XScale','log')
set(gca,'YScale','log')
xlabel('Number of nodes $n$')
ylabel('Mean relative quadrature error')
errors = all_errors{3};
means = mean(errors,2);
plot(ns,means(end) * ns.^(-5/2) / ns(end)^(-5/2),'k--')
legend({'Monte Carlo','IID','RPCholesky (Ours)'...
    '$\mathcal{O}(n^{-5/2})$'},'Location','Southwest')
axis([-Inf Inf 1e-4 4])