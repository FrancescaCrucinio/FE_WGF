using Distributions;
using Plots;
include("drift_exact.jl")
include("drift_approximate.jl")

x = collect(range(0, 1, length = 1000));
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
sigma0 = 0.01;
M = 5;
g(x, y) = (y .- x)./sigmaG .* pdf.(Normal.(x, sqrt(sigmaG)), y);
y = rand(Normal(0.5, sqrt(sigmaH)), M);
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
p1 = plot(f, 0, 1, legend = false)
title!("f")
gradient = zeros(M, 1000);
p2 = plot()
for i=1:M
    gradient[i, :] = g(x, y[i]);
    plot!(p2, x, gradient[i, :])
end
title!("gradient")
driftE = drift_exact(0.5, sigma0, sigmaG, sigmaH, x);
p3 = plot(x, driftE)
title!("exact drift")
driftA = drift_approximate(0.5, sigma0, sigmaG, sigmaH, x, 1000000);
p4 = plot(x, driftA)
title!("approximate drift")
plot(p1, p2, p3, p4, layout=(4, 1))

plot(p3, p4, layout=(1,2))
