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

N = 5;
M = 100;
Niter = 100;
lambda = 0.01;
x = AT_wgf(N, M, Niter, lambda);
figure(1);
histogram(x(Niter, :))

figure(2);
fplot(f, [0, 1], '-k', 'Linewidth', 4)
hold on
[KDEy, KDEx] = ksdensity(x(Niter, :), 'Function', 'pdf');
plot(KDEx, KDEy)