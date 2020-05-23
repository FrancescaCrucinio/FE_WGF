# plot for parameter α selection
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using JLD;
using LaTeXStrings;


lambda = load("analytically tractable/parameters1000.jld", "lambda");
diagnosticsWGF = load("analytically tractable/parameters1000.jld", "diagnosticsWGF");

pyplot()
p1 = plot(lambda, diagnosticsWGF[:, 1], lw = 3, legend = false,
        xlabel="lambda", ylabel="mean");
hline!([0.5]);
p2 = plot(lambda, diagnosticsWGF[:, 2], lw = 3, legend = false,
        xlabel="lambda", ylabel="variance");
hline!([0.043^2]);
p3 = plot(lambda, diagnosticsWGF[:, 3], lw = 3, legend = false,
        xlabel="lambda", ylabel="95th MSE");
p4 = plot(lambda, diagnosticsWGF[:, 4], lw = 3, legend = false,
        xlabel="lambda", ylabel="MISE");
p5 = plot(lambda, diagnosticsWGF[:, 5], lw = 3, legend = false,
        xlabel="lambda", ylabel="E(rho)");
p6 = plot(lambda, diagnosticsWGF[:, 6], lw = 3, legend = false,
        xlabel="lambda", ylabel="entropy");

p7 = plot(lambda, diagnosticsWGF[:, 7], lw = 3, legend = false,
        xlabel="lambda", ylabel="kl");

plot(p1, p2, p3, p4, p5, p6, layout = (2, 3))

# savefig(p1, "mean1000.pdf")
# savefig(p2, "var1000.pdf")
# savefig(p3, "mse1000.pdf")
# savefig(p4, "mise1000.pdf")
# savefig(p5, "e1000.pdf")
# savefig(p6, "entropy1000.pdf")
