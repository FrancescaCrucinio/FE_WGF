module smcems

using Distributions;
using Statistics;
using StatsBase;

using samplers;

export smc_gaussian_mixture
export optimal_bandwidthESS
export smc_p_dim_gaussian_mixture

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
INPUTS
'N' number of particles
'Niter' number of time steps
'epsilon' standard deviation for Gaussian smoothing kernel
'x0' initial distribution.
'muSample' sample from μ
'sigmaK' standard deviation of k
=#
function smc_p_dim_gaussian_mixture(N, Niter, epsilon, x0, muSample, sigmaK)
    # initial distribution
    xOld = x0;
    # number of dimensions
    p = size(x0, 2);
    # uniform weights at time n = 1
    W = ones[N, 1]/N;
    # number of samples to draw from μ(y)
    M = min(N, size(muSample, 1));
    for n=2:Niter
        # sample from μ
        yIndex = sample(1:size(hSample, 1), M, false);
        y = muSample[yIndex, :];
        # ESS
        ESS=1/sum(W.^2);
        # RESAMPLING
        if(ESS < N/2)
            indices = trunc.(Int, mult_resample(W, N));
            xNew = xOld[indices];
            W .= 1/N;
        else
            xNew = xOld;
        end

        # Markov kernel
        xNew = xNew + epsilon*randn(N, p);

        # Compute μ^N_{n}
        muN = zeros(M, 1);
        for j=1:M
            muN[j] = mean(W .* pdf(MvNormal(y[j, :], sigmaK*Matrix{Float64}(I, 2, 2)), xNew'));
        end

        # update weights
        for i=1:N
            g = pdf(MvNormal(xNew[i, :], sigmaK*Matrix{Float64}(I, 2, 2)), y'));
            # potential at time n
            potential = mean(g ./ hN);
            # update weight
            W[i] = W[i] * potential;
        end
        # normalise weights
        W = W / sum(W);
        xOld = xNew;
    end
end
end
