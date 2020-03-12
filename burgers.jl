using Distributions;
using Plots;
using Random;

N = 1000;
dt = 1e-02;
Niter = trunc(Int, 1/dt);
sigma = 0.1;

heaviside(t) = 0.5 * (sign(t) + 1);

x = zeros(Niter, N);
x[1, :] = rand(1, N);
for n=1:(Niter-1)
    drift = zeros(N, 1);
    for i=1:N
        drift[i] = mean(heaviside.(x[n, i] .- x[n, :]));
    end
    x[n+1, :] = x[n, :] - drift*dt + sigma *rand(Normal(0, dt), N, 1);
end
