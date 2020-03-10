using Distributions
using Random
using Plots

N = 10;
M = 1000;
Niter = 10^5;
lambda = 1;
dt = 1/Niter;

x = zeros(Niter, N);
drift = zeros(Niter, N);
y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
x[1, :] = rand(1, N);
for n=1:(Niter-1)
    hN = zeros(M, 1);
    for j=1:M
        hN[j] = sum(pdf.(Normal.(x[n, :], 0.045), y[j]));
    end

    for i=1:N
        gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
        drift[n, i] = sum(gradient./hN);
    end
    x[n+1, :] = x[n, :] .- drift[n, :]*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
end

f(x) = pdf.(Normal(0.5, 0.043), x);
plot(f, 0, 1)
histogram(x[Niter, :])
