module smcems

using Distributions;
using Statistics;
using StatsBase;
using LinearAlgebra;

using samplers;

export smc_gaussian_mixture
export optimal_bandwidthESS
export smc_mixture_hd
export mixture_hd_kde_weighted

#= SMC for gaussian mixture exmaple
OUTPUTS
1 - particle locations
2 - particle weights
INPUTS
'N' number of particles
'Niter' number of time steps
'epsilon' standard deviation for Gaussian smoothing kernel
'x0' initial distribution.
user selected initial distribution
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
=#
function smc_gaussian_mixture(N, Niter, epsilon, x0, muSample, M)
    # initialise a matrix x storing the particles at each time step
    x = zeros(Niter,N);
    # initialise a matrix W storing the weights at each time step
    W = zeros(Niter,N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # uniform weights at time n = 1
    W[1, :] = ones(1, N)/N;

    for n=2:Niter
        # samples from μ(y)
        y = sample(muSample, M, replace = false);
        # ESS
        ESS=1/sum(W[n-1,:].^2);
        # RESAMPLING
        if(ESS < N/2)
            indices = trunc.(Int, mult_resample(W[n-1,:], N));
            x[n,:] = x[n-1, indices];
            W[n,:] .= 1/N;
        else
            x[n,:] = x[n-1,:];
            W[n,:] = W[n-1,:];
        end

        # Markov kernel: Random walk step
        x[n, :] = x[n, :] + epsilon*randn(N, 1);

        # Compute μ^N_{n}
        hN = zeros(M,1);
        for j=1:M
            hN[j] = mean(W[n, :] .* pdf.(Normal.(x[n, :], 0.045), y[j]));
        end

        # update weights
        for i=1:N
            K = pdf.(Normal.(x[n, i], 0.045), y);
            potential = mean(K ./ hN);
            # update weight
            W[n, i] = W[n, i] * potential;
        end
        # normalise weights
        W[n, :] = W[n, :] ./ sum(W[n, :]);
    end
    return x, W
end

#= Multinomial resampling
OUTPUTS
1- indices of resampled particles
INPUTS
'W' vector of weights
'N' number of particles to sample (in most cases N = length(W))
=#
function mult_resample(W, N)
    # vector to store number of offsprings
    indices = zeros(N, 1);
    # start inverse transfor method
    s = W[1];
    u = sort(rand(N, 1), dims = 1);
    j = 1;
    for i=1:N
    while(s < u[i])
      j = j+1;
      s = s + W[j];
    end
    indices[i] = j;
    end
    return indices
end

#=
 Optimal bandwidth for Gaussian kernel with effective sample size (ESS)
OUTPUTS
1 - bandwidth
INPUTS
'x' particle locations
'W' particle weights
=#
function optimal_bandwidthESS(x, W)
    # ESS
    ESS = sum(W.^2);
    # standard deviation (weighted)
    s = Statistics.std(x, weights(W), corrected = false);
    # bandwidth
    bw = 1.06*s*ESS^(1/5);

    return bw
end

#= SMC-EMS for gaussian mixture exmaple -- d>1
OUTPUTS
1 - particle locations
2 - particle weights
3 - value of KL
INPUTS
'N' number of particles
'Niter' number of time steps
'epsilon' standard deviation for Gaussian smoothing kernel
'x0' initial distribution.
'muSample' sample from μ
'sigmaK' standard deviation of k
'funtional' if true the value of the KL divergence at each iteration is returned
=#
function smc_mixture_hd(N, Niter, epsilon, x0, muSample, sigmaK, functional)
    # initial distribution
    x = copy(x0);
    # number of dimensions
    d = size(x0, 1);
    # uniform weights at time n = 1
    W = ones(N)/N;
    # number of samples to draw from μ(y)
    M = min(N, size(muSample, 2));
    E = zeros(Niter);
    for n=2:Niter
        # sample from μ
        yIndex = sample(1:size(muSample, 2), M, replace = false);
        y = muSample[:, yIndex];
        # ESS
        ESS=1/sum(W.^2);
        # RESAMPLING
        if(ESS < N/2)
            indices = trunc.(Int, mult_resample(W, N));
            x = x[:, indices[:]];
            W .= 1/N;
        end

        # Markov kernel
        x = x .+ epsilon*randn(d, N);

        # Compute μ^N_{n}
        muN = zeros(M);
        for j=1:M
            muN[j] = mean(W .* pdf(MvNormal(y[:, j], sigmaK^2*Matrix{Float64}(I, d, d)), x));
        end

        if(functional)
            # log-likelihood
            loglik = -log.(muN);
            E[n-1] =  mean(loglik);
        end

        # update weights
        for i=1:N
            g = pdf(MvNormal(x[:, i], sigmaK^2*Matrix{Float64}(I, d, d)), y);
            # potential at time n
            potential = mean(g ./ muN);
            # update weight
            W[i] = W[i] * potential;
        end
        # normalise weights
        W = W / sum(W);
    end
    # sample from μ
    yIndex = sample(1:size(muSample, 2), M, replace = false);
    y = muSample[:, yIndex];
    # Compute μ^N_{n}
    muN = zeros(M, 1);
    for j=1:M
        muN[j] = mean(W .* pdf(MvNormal(y[:, j], sigmaK^2*Matrix{Float64}(I, 2, 2)), x));
    end
    if(functional)
        # log-likelihood
        loglik = -log.(muN);
        E[Niter] =  mean(loglik);
    end
    return x, W, E
end
#= Kernel density estimatior for mixture model in d dimension
OUTPUTS
1 - KDE evaluated at KDEeval
INPUTS
'piSample' sample from π (dxN matrix)
'W' weights for piSample
'KDEeval' evaluation points (d rows matrix)
'epsilon' standard deviation for Gaussian smoothing kernel
=#
function mixture_hd_kde_weighted(piSample, W, KDEeval, epsilon)
    # dimension
    d = size(piSample, 1);
    # number of samples
    N = size(piSample, 2);
    # Silverman's plug in bandwidth
    bw = zeros(d);
    for i=1:d
        bw[i] = sqrt(epsilon^2 + optimal_bandwidthESS(piSample[i, :], W)^2);
    end
    # kde
    KDEdensity = zeros(size(KDEeval, 1));
    for i = 1:size(KDEeval, 1)
        KDEdensity[i] = sum(W.*pdf(MvNormal(KDEeval[i, :], Diagonal(bw.^2)), piSample))/prod(bw);
    end
    return KDEdensity;
end

end
