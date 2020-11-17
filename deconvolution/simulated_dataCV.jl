# push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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
using wgf;

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

a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = @rget xx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = @rget muKDE;
    refY = @rget xx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
alpha = range(0.05, stop = 0.2, length = 10);
Nparticles = 500;
dt = 1e-2;
Niter = 1000;
M = 500;
x0 = sample(muSample, Nparticles, replace = true);
# divide muSample into groups
L = 5;
muSample = reshape(muSample, (L, Int(length(muSample)/L)));

E = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        # WGF
        x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha[i], x0, muSampleL, M, 0.5, sigU);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
