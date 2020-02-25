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

N = 100;
M = 100;
Niter = 100;
lambda = 0.01;
x = zeros(Niter, N);

x(1, :) = rand(1, N);
for n=1:Niter
    % get samples from h(y)
    y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(M, 1);
    % Compute h^N_{n}
    hN = zeros(M,1);
    for j=1:M
        hN(j) = sum(normpdf(y(j), x(n, :), 0.045));
    end
    drift = zeros(1, N);
    for i=1:N
        gradient = normpdf(y, x(n, i), 0.045) .* (y - x(n, i))/(0.045^2);
        drift(i) = sum(gradient./hN);
    end
    x(n+1, :) = x(n, :) + drift/Niter + sqrt(2*lambda)*randn(1, N);
end

[KDEy, KDEx] = ksdensity(x(Niter, :), 'Function', 'pdf');

fplot(f, [0, 1], '-k', 'Linewidth', 4)
hold on
plot(KDEx, KDEy)