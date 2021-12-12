push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using KernelEstimator;
using DelimitedFiles;

means = [0.3 0.7];
variances = [0.07^2; 0.1^2];
# true mean, variance and probability of quadrant
m = means[1]/3 + 2*means[2]/3;
v = variances[1]/3 + 2*variances[2]/3 + means[1]^2/3 + 2*means[2]^2/3 - m^2;

Nrep = 100;
dims = 10;
tSMC1000 = zeros(Nrep, dims);
tWGF1000 = zeros(Nrep, dims);
entWGF1000 = zeros(Nrep, dims);
entSMC1000 = zeros(Nrep, dims);
mWGF1000 = zeros(Nrep, dims);
mSMC1000 = zeros(Nrep, dims);
vWGF1000 = zeros(Nrep, dims);
vSMC1000 = zeros(Nrep, dims);
pWGF1000 = zeros(Nrep, dims);
pSMC1000 = zeros(Nrep, dims);
for i in 1:dims
    p = ((cdf(Normal(means[1], variances[1]), 0.5) - cdf(Normal(means[1], variances[1]), 0) +
        2*(cdf(Normal(means[2], variances[2]), 0.5) - cdf(Normal(means[2], variances[2]), 0)))/3)^i;
    readf = readdlm("mixture_hd/1000smc_vs_wgf_$i.txt", ',', Float64);
    tSMC1000[:, i] = readf[:, 1];
    mSMC1000[:, i] = readf[:, 2]/m;
    vSMC1000[:, i] = readf[:, 3]/v;
    pSMC1000[:, i] = readf[:, 4]/p;
    tWGF1000[:, i] = readf[:, 5];
    mWGF1000[:, i] = readf[:, 6]/m;
    vWGF1000[:, i] = readf[:, 7]/v;
    pWGF1000[:, i] = readf[:, 8]/p;
end
# m,v,p and time vs dims
p1 = plot(1:dims, mean(mSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 15)
plot!(1:dims, mean(mWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1", ylims = (1e-07, 1e-04))
# savefig(p1, "mixture_hd_means.pdf")

p2 = plot(1:dims, mean(vSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(vWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1", ylims = (1e-07, 1e-04))
# savefig(p2, "mixture_hd_variances.pdf")

p3 = plot(1:dims, mean(pSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(pWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1")
# savefig(p3, "mixture_hd_probs.pdf")

p4 = plot(1:dims, mean(tSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(tWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1")
# savefig(p4, "mixture_hd_times2.pdf")
