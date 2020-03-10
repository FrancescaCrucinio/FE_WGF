using Distributions
using Random
using Plots

sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;

N = 10;
M = 1000;
dt = 1e-02;
lambda = 0.01;
Niter = trunc(Int, 1/dt);

x = zeros(Niter, N);
variance = zeros(Niter, N);
drift = zeros(Niter, N);
variance[1, :] = sigma0;
drift[1, :] = 0.5;
y = rand(Normal(0.5, sqrt(sigmaH)), M);
x[1, :] = rand(1, N);
for n=1:(Niter-1)
    for i=1:N
        drift[n+1] = drift_exact(drift[n], variance[n], sigmaG, sigmaH, x[n, :]);
    x[n+1, :] = x[n, :] .- drift[n+1].*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    drift[n+1] = drift[n] .- drift[n+1].*dt;
    variance[n+1] = variance[n] + 2*lambda/dt^2;
end

f(x) = pdf.(Normal(0.5, 0.043), x);
plot(f, 0, 1)
histogram(x[100, :])
