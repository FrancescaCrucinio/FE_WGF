% set seed
rng('default');

% variances for f, g, h
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaG + sigmaF;
% f, g, h are Normals
h = @(y) normpdf(y, 0.5, sqrt(sigmaH));
g = @(x,y) normpdf(y, x, sqrt(sigmaG));
f = @(x) normpdf(x, 0.5, sqrt(sigmaF));

N = 50000;
M = 1000;
Niter = 100;
lambda = 0.01;
x = AT_wgf_exact(N, Niter, lambda);

figure(1);
subplot(1,2,1)
histogram(x(Niter, :))
subplot(1,2,2)
[KDEy, KDEx] = ksdensity(x(Niter, :), 'Function', 'pdf');
plot(KDEx, KDEy)

figure(2);
fplot(f, [0, 1], '-k', 'Linewidth', 4)
hold on
plot(KDEx, KDEy)

mean(x(Niter, :))
var(x(Niter, :), 1)
sigmaF