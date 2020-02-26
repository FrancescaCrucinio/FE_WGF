function [x]= burgers(N, Niter, sigma)

x = zeros(Niter, N);
for n=1:Niter
    drift = zeros(1, N);
    for i=1:N
        drift(i) = mean(heaviside(x(n, i) - x(n, :)));
    end
    x(n+1, :) = x(n, :) - drift/Niter + sigma * sqrt(1/Niter)*randn(1, N);
end
end