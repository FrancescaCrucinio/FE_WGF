using Distributions
using Random
using Plots

N = 100;
M = 1000;
Niter = 10^2;
lambda = 0.01;
dt = 1/Niter;

x = zeros(Niter, N);
y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
x[1, :] = rand(1, N);
for n=1:(Niter-1)
    hN = zeros(M, 1);
    for j=1:M
        hN[j] = sum(pdf.(Normal.(x[n, :], 0.045), y[j]));
    end
    drift = zeros(N, 1);
    for i=1:N
        gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
        drift[i] = sum(gradient./hN);
    end
    x[n+1, :] = x[n, :] .- drift*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
end

f(x) = pdf.(Normal(0.5, 0.043), x);
plot(f, 0, 1)
histogram(x[100, :])
