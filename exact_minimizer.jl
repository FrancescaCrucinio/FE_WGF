
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


sigma = range(0, stop = 1, length = 1000);
kl_entropy(sigma, l) = log.((sigmaG .+ sigma)/sigmaH)/2 .+
    0.5*sigmaH./(sigmaG .+ sigma) .- 0.5 .- 0.5*l*(1 .+ log.(2*pi*sigma));

lvalues =  range(0, stop = 0.99, length = 10);
M = length(lvalues);
target = zeros(M, 1000);
p = plot()
for i=1:M
    target[i, :] = kl_entropy(sigma, lvalues[i]);
    plot!(p, sigma, target[i, :]);
end
plot(p)
