push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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

Nparticles = load("analytically tractable/comparison_uniform17062020.jld", "Nparticles");
tSMC = load("analytically tractable/comparison_uniform17062020.jld", "tSMC");
tWGF = load("analytically tractable/comparison_uniform17062020.jld", "tWGF");
diagnosticsSMC = load("analytically tractable/comparison_uniform17062020.jld", "diagnosticsSMC");
diagnosticsWGF = load("analytically tractable/comparison_uniform17062020.jld", "diagnosticsWGF");
# qSMC = load("analytically tractable/comparison_uniform17062020.jld", "qdistSMC");
# qWGF = load("analytically tractable/comparison_uniform17062020.jld", "qdistWGF");

# markers = [:circle :rect :diamond :xcross];
markers = [:circle :rect :diamond :xcross :star5];
# labels = ["N=100" "N=500" "N=1000" "N=5000"];
labels = ["N=100" "N=500" "N=1000" "N=5000" "N=10000"];
pyplot()
p = plot([tSMC tWGF], [diagnosticsSMC[:, 3] diagnosticsWGF[:, 3]],
    lw = 3, label = ["SMC" "WGF"], xlabel="Runtime (s)", ylabel="MISE",
    xaxis=:log, color = [1 2], ylims = (-1.5, +Inf));
for i=1:length(tSMC)
    scatter!(p, [tSMC[i]], [diagnosticsSMC[i, 3]],
        markersize = 9, label = labels[i], color = :black, marker = markers[i],
        markerstrokewidth=0);
    scatter!(p, [tSMC[i] tWGF[i]], [diagnosticsSMC[i, 3] diagnosticsWGF[i, 3]],
        markersize = 9, label = ["" ""], color = [1 2], marker = markers[i],
        markerstrokewidth=0);
end
p
# # savefig(p1, "comparison_runtime.pdf")
# # savefig(p2, "comparison_mean.pdf")
# # savefig(p3, "comparison_var.pdf")
# # savefig(p4, "comparison_mise.pdf")
