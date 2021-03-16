push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
using JLD;
@rimport ks as rks;
# custom packages
using wgf_prior;
R"""
library(fDKDE)
library(tictoc)
"""

# set seed
Random.seed!(1234);

# grid over support
KDEx = range(-2, stop = 8, step = 0.1);
dx = KDEx[2] - KDEx[1];
true_density = pdf.(MixtureModel(Normal, [(5, 0.4), (2, 1)], [0.25, 0.75]), KDEx);
# parameters for WGF
# number of particles
Nparticles = [100 500 1000 5000];
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 500;
# regularisation parameter
alpha = 0.1;

# synthetic data
# noise to signal ratio
NSR = 0.2;


Nrep = 10;
isePI = zeros(length(Nparticles), Nrep);
iseCV = zeros(length(Nparticles), Nrep);
iseWGF = zeros(length(Nparticles), Nrep);
timePI = zeros(length(Nparticles), Nrep);
timeCV = zeros(length(Nparticles), Nrep);
timeWGF = zeros(length(Nparticles), Nrep);
for i=1:length(Nparticles)
    for j=1:Nrep
        # data from a normal mixture
        true_dataDKDE = rand(MixtureModel(Normal, [(5, 0.4), (2, 1)], [0.25, 0.75]), Nparticles[i], 1);
        error_sdDKDE = sqrt(NSR*var(true_dataDKDE));
        muSampleDKDE = true_dataDKDE .+ error_sdDKDE*randn(Nparticles[i], 1);
        true_dataWGF = rand(MixtureModel(Normal, [(5, 0.4), (2, 1)], [0.25, 0.75]), 10^3, 1);
        error_sdWGF = sqrt(NSR*var(true_dataWGF));
        muSampleWGF = true_dataWGF .+ error_sdWGF*randn(10^3, 1);

        # DKDEpi & DKDEcv
        R"""
        # PI bandwidth of Delaigle and Gijbels
        tic()
        hPI <- PI_deconvUknownth4(c($muSampleDKDE), "norm", $error_sdDKDE^2, $error_sdDKDE);
        fdec_hPI <- fdecUknown($KDEx, c($muSampleDKDE), hPI, "norm", $error_sdDKDE, $dx);
        exectime <- toc()
        exectimePI <- exectime$toc - exectime$tic

        tic()
        hCV <- CVdeconv(c($muSampleDKDE), "norm", $error_sdDKDE);
        fdec_hCV <-  fdecUknown($KDEx, c($muSampleDKDE), hCV, "norm", $error_sdDKDE, $dx);
        exectime <- toc()
        exectimeCV <- exectime$toc - exectime$tic
        """
        # runtimes and ise
        timePI[i, j] = @rget exectimePI;
        timeCV[i, j] = @rget exectimeCV;
        isePI[i, j] = dx*sum((true_density .- @rget(fdec_hPI)).^2);
        iseCV[i, j] = dx*sum((true_density .- @rget(fdec_hCV)).^2);

        # WGF
        # prior mean = mean of μ
        m0 = mean(muSampleWGF);
        sigma0 = std(muSampleWGF);
        # initial distribution
        M = min(Nparticles[i], 10^3);
        x0 = sample(muSampleWGF, Nparticles[i], replace = true);
        timeWGF[i, j] = @elapsed begin
        x = wgf_DKDE_tamed(Nparticles[i], dt, Niter, alpha, x0, m0, sigma0, muSampleWGF, M, error_sdWGF);
        RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
        KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
        end
        iseWGF[i, j] = dx*sum((true_density .- KDEyWGF).^2);
    end
end

runtimeWGF = mean(timeWGF, dims = 2);
runtimePI = mean(timePI, dims = 2);
runtimeCV = mean(timeCV, dims = 2);
errorWGF = mean(iseWGF, dims = 2);
errorPI = mean(isePI, dims = 2);
errorCV = mean(iseCV, dims = 2);

p = plot(runtimePI, errorPI, yaxis = :log, xaxis = :log, lw = 3, color = :gray, line = :dashdot, label = "DKDEpi",
    legendfontsize = 10, tickfontsize = 10, legend = :outerright)
plot!(p, runtimeCV, errorCV, xaxis = :log, lw = 3, color = :blue, line = :dot,  label = "DKDEcv")
plot!(p, runtimeWGF, errorWGF, xaxis = :log, lw = 3, color = :red, line = :solid, label = "WGF")
markers = [:circle :rect :diamond :star5 :xcross];
for i=1:length(Nparticles)
    N = Nparticles[i];
    scatter!(p, [runtimePI[i]], [errorPI[i]], xaxis = :log, color = :black,
        markerstrokecolor = :black, marker = markers[i], markersize = 5, label = "N = $N")
    scatter!(p, [runtimePI[i]], [errorPI[i]], xaxis = :log, color = :gray,
        markerstrokecolor = :gray, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [runtimeCV[i]], [errorCV[i]], xaxis = :log, color = :blue,
        markerstrokecolor = :blue, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [runtimeWGF[i]], [errorWGF[i]], xaxis = :log, color = :red,
        markerstrokecolor = :red, marker = markers[i], markersize = 5, label = "")
end

using JLD;
save("deconv_rate14Mar2021.jld", "timeWGF", timeWGF, "timeCV", timeCV, "timePI", timePI,
     "iseWGF", iseWGF, "iseCV", iseCV, "isePI", isePI);
