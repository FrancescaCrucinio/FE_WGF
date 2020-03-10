using Distributions;
using Plots;

N = 10000;
x = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), N);
f(x) = pdf.(Normal(0.5, 0.043), x);
plot(f, 0, 1)

histogram(x)
