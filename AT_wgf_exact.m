function [x]= AT_wgf_exact(N, Niter, lambda)

sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaG + sigmaF;
sigma0 = rand(1);

x = zeros(Niter, N);
x(1, :) = 0.5 + sigma0 * randn(1, N);
dt = 1/Niter;
variance = zeros(Niter, N);
drift = zeros(Niter, N);
variance(1, :) = sigma0;
drift(1, :) = 0.5;
for n=2:Niter
    for i=1:N
        [variance(n, i), drift(n, i)] = exactAT_wgf(variance(n-1, i), x(n, i),...
            drift(n-1, i), lambda, sigmaG, sigmaH, dt);
        x(n+1, i) = drift(n, i) + sqrt(variance(n, i))*randn(1);
    end
end
end