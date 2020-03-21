
using Distributions;
using Plots;

sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;

lambda = [range(0, stop=0.9, length = 10); range(1.1, stop = 10, length = 1000)];
variance  = (sigmaH - sigmaG .+ 2*lambda*sigmaG +
            sqrt.(sigmaG^2 + sigmaH^2 .- 2*sigmaG*sigmaH.*(1 .- 2*lambda)))./
            (2*(1 .- lambda));

plot(lambda, variance)
hline!([sigmaF])
