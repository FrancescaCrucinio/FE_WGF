push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using XLSX;
using RCall;
@rimport ks as rks
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# remove non finite elements for entropy computation
function remove_non_finite(x)
       return isfinite(x) ? x : 0
end

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 200000;
# number of particles
Nparticles = 500;
# number samples from μ
M = 500;
# regularisation parameters
alpha = range(0.1, stop = 50, length = 10);

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);


# sample from μ
muSample = float.(XLSX.readdata("sucrase_Carter1981.xlsx", "Foglio1!B2:B25"));
muSample = muSample[:];
# number of sub-groups for CV
L = size(muSample, 1);
# KDE for μ
bw = 1.06*std(muSample)*length(muSample)^(-1/5);
RKDE = rks.kde(muSample, var"h" = bw);
muKDEy = abs.(rcopy(RKDE[3]));
# reference values for KL divergence
muKDEx = rcopy(RKDE[2]);
# estimate error variance from data
sigma = sqrt(0.25*var(muSample));

# initial distribution
x0 = sample(muSample, M, replace = true);

E = zeros(length(alpha), L);
Threads.@threads for i=1:length(alpha)
    @simd for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        # WGF
        x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5);
        # KDE
        RKDE = rks.kde(x[Niter, :], var"eval.points" = KDEx);
        KDEy = abs.(rcopy(RKDE[3]));
        ent = -mean(remove_non_finite.(KDEy .* log.(KDEy)));
        # approximated value
        delta = muKDEx[2] - muKDEx[1];
        hatH = zeros(1, length(muKDEx));
        # convolution with approximated f
        # this gives the approximated value
        for i=1:length(muKDEx)
            hatH[i] = delta*sum(pdf.(Normal.(KDEx, sigma), muKDEx[i]).*KDEy);
        end
        kl = kl_divergence(muKDEy, hatH);
        E[i, l] = kl-alpha[i]*ent;
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
