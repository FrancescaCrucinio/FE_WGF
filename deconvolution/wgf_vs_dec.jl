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
# custom packages
using wgf_prior;

R"""
library(tictoc)
library(fDKDE)
library(ks)
library(ggplot2)
# set seed
set.seed(1234);

# parameters for dec
#Noise to signal ratio=varU/varX
NSR=0.2

#Sample size
n=500

# grid over support
muKDEx <- seq(-2,8,0.1);
"""
# set seed
Random.seed!(1234);
# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-3;
# number of iterations
Niter = 1000;
# regularisation parameter
alpha = 0.011;

Nrep = 100;
ise = zeros(3, Nrep);
runtime = zeros(3, Nrep);
for i=1:Nrep
    R"""
    #Generate data from a normal mixture
    X=rnorm(n,5,.4);
    X2=matrix(rnorm(n*n,2,1),nrow=n,ncol=n,byrow=TRUE);

    pmix=0.75;
    tmp=matrix(runif(n,0,1),nrow=1,ncol=n,byrow=TRUE);
    X[which(tmp<pmix)]=X2[which(tmp<pmix)];

    # true density
    tdensity <- truedens(muKDEx);

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

    #DKDE estimators
    dx = muKDEx[2] - muKDEx[1];
    #PI bandwidth of Delaigle and Gijbels
    tic()
    hPI=PI_deconvUknownth4(W,errortype,varU,sigU);
    fdec_hPI = fdecUknown(muKDEx,W,hPI,errortype,sigU,dx);
    exectime <- toc()
    exectimePI <- exectime$toc - exectime$tic
    isePI = var(tdensity - fdec_hPI);

    tic()
    hCV=CVdeconv(W,errortype,sigU);
    fdec_hCV = fdecUknown(muKDEx,W,hCV,errortype,sigU,dx);
    exectime <- toc()
    exectimeCV <- exectime$toc - exectime$tic
    iseCV = var(tdensity - fdec_hCV);
    """
    # runtimes and ise
    runtime[1, i] = @rget exectimePI;
    runtime[2, i] = @rget exectimeCV;
    ise[1, i] = @rget isePI;
    ise[2, i] = @rget iseCV;
    # get sample from μ
    muSample = @rget W;
    # prior mean = mean of μ
    m0 = mean(muSample);
    sigma0 = std(muSample);
    # get parameter for K
    sigU = @rget sigU;
    # initial distribution
    x0 = sample(muSample, Nparticles, replace = true);

    runtime[3, i] = @elapsed begin
    x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, 0.5, sigU);
    end
    # KDE
    R"""
    KDE_wgf <- kde($x[$Niter, ], eval.points = muKDEx);
    iseWGF = var(tdensity - KDE_wgf$estimate);
    """
    ise[3, i] = @rget iseWGF;
    println("$i")
end

mean(ise, dims = 2)
times = mean(runtime, dims = 2);

tpi = times[1];
tcv = times[2];
tWGF = times[3];

distpi = ise[1, :];
distcv = ise[2, :];
distWGF = ise[3, :];
# plot
R"""
    library(ggplot2)
    g <- rep(1:3, each= 100)
    runtime <- rep(c($tpi, $tcv, $tWGF), each = 100)
    runtime <- round(runtime, 2)
    data <- data.frame(x = runtime, y = c($distpi, $distcv, $distWGF))
    data$g <- factor(g, labels = c("DKDEpi", "DKDEcv", "WGF"))
    p <- ggplot(data, aes(x = x, y = y)) +
    geom_boxplot(lwd = 1, aes(x = x, y = y, fill = g), width = 8) +
    scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x), labels = scales::trans_format("log10", scales::math_format(10^.x))) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(),
        aspect.ratio = 2/3, legend.key.size = unit(1, "cm"), plot.margin=grid::unit(c(0,0,0,0), "mm"))
    # ggsave("dkde_vs_wgf.eps", p, height = 4)
"""
