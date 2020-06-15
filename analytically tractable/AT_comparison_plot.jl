push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# push!(LOAD_PATH, "/homes/crucinio/WGF/myModules")
# Julia packages
# using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using LaTeXStrings;

Nparticles = load("analytically tractable/comparison_delta.jld", "Nparticles");
tSMC = load("analytically tractable/comparison_delta.jld", "tSMC");
tWGF = load("analytically tractable/comparison_delta.jld", "tWGF");
diagnosticsSMC = load("analytically tractable/comparison_delta.jld", "diagnosticsSMC");
diagnosticsWGF = load("analytically tractable/comparison_delta.jld", "diagnosticsWGF");

p1 = plot(Nparticles, [tSMC tWGF], lw = 3, xlabel="N", ylabel="Runtime",
        label = ["SMC" "WGF"], legend=:topleft);
p2 = plot(Nparticles, [diagnosticsSMC[:, 1] diagnosticsWGF[:, 1]],
    lw = 3, xlabel="N", ylabel="mean", legend = false);
hline!([0.5]);
p3 = plot(Nparticles, [diagnosticsSMC[:, 2] diagnosticsWGF[:, 2]],
    lw = 3, legend = false, xlabel="N", ylabel="variance");
hline!([0.043^2]);
p4 = plot(Nparticles, [diagnosticsSMC[:, 3] diagnosticsWGF[:, 3]],
    lw = 3, legend = false, xlabel="N", ylabel="MISE");
plot(p1, p2, p3, p4, layout = (2, 2))
#
# # savefig(p1, "comparison_runtime.pdf")
# # savefig(p2, "comparison_mean.pdf")
# # savefig(p3, "comparison_var.pdf")
# # savefig(p4, "comparison_mise.pdf")
