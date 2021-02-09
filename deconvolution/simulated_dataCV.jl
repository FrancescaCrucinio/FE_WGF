push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf_prior;

# set seed
Random.seed!(1234);

# R example
R"""
library(tictoc)
library(fDKDE)
#Noise to signal ratio=varU/varX
NSR=0.2

#Sample size
n=500

#Generate data from a normal mixture
X=rnorm(n,5,.4);
X2=matrix(rnorm(n*n,2,1),nrow=n,ncol=n,byrow=TRUE);

pmix=0.75;
tmp=matrix(runif(n,0,1),nrow=1,ncol=n,byrow=TRUE);
X[which(tmp<pmix)]=X2[which(tmp<pmix)];

#Specify error distribution (normal or Laplace in this case) and generate data from this error distribution
errortype="Lap";
#normal case
if (errortype=="norm")
{sigU=sqrt(NSR*var(X));
U=rnorm(n,0,sigU);
varU=sigU^2;}

#Laplace case
if (errortype=="Lap")
{sigU=sqrt(NSR*var(X)/2);
varU=2*sigU^2;
U=rlap(sigU,1,n);}

#Contaminated data
W=as.vector(X+U);

# DKDE
# grid over support
xx <- seq(-2,8,0.1);
# true density
tdensity <- truedens(xx);
# Delaigle's estimators
tic()
outcome<-fDKDE(W,errortype,NSR,n,varU,sigU);
muKDE <- outcome$naive_KDE;
toc()
names(outcome)<-c('PI_bandwidth','DKDE_nonrescaledPI','DKDE_rescaledPI','CV_bandwidth','DKDE_rescaledCV','normal_bandwidth','naive_KDE')
"""

# functional approximation
function psi(piSample, a, m0, sigma0)
    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(pdf.(Laplace.(muSample[i], sigU), piSample));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(Normal(m0, sigma0), piSample);
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    kl_prior = mean(prior./pihat .- 1 .- log.(prior./pihat));
    return kl+a*kl_prior;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 500;
# regularisation parameter
alpha = range(0.0001, stop = 0.005, length = 10);

# divide muSample into groups
L = 5;
muSample = reshape(muSample, (L, Int(length(muSample)/L)));

E = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        muSampleL = muSampleL[:];
        # initial distribution
        x0 = sample(muSampleL, M, replace = true);
        # prior mean = mean of μ
        m0 = mean(muSampleL);
        sigma0 = std(muSampleL);
        # WGF
        x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSampleL, M, 0.5, sigU);
        # functional
        E[i, l] = psi(x[Niter, :], alpha[i], m0, sigma0);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
