module wgf_prior;

using Distributions;
using Statistics;
using LinearAlgebra;

using samplers;

export wgf_flu_tamed
export wgf_DKDE_tamed
export wgf_ct_tamed
export wgf_ct_tamed_cv
export wgf_hd_mixture_tamed

#= WGF for Spanish flu data
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
=#
function wgf_flu_tamed(N, dt, Niter, alpha, x0, m0, sigma0, muSample, M)
   # initialise a matrix x storing the particles
   x = zeros(Niter, N);
   # initial distribution is given as input:
   x[1, :] = x0;

   for n=1:(Niter-1)
       # get samples from μ(y)
       y = sample(muSample, M, replace = false);
       # Compute denominator
       muN = zeros(M, 1);
       for j=1:M
           muN[j] = mean(0.595*pdf.(Normal(8.63, 2.56), y[j] .- x[n, :]) +
                   0.405*pdf.(Normal(15.24, 5.39), y[j] .- x[n, :]));
       end

       # gradient and drift
       drift = zeros(N, 1);
       for i=1:N
           gradient = 0.595*pdf.(Normal(8.63, 2.56), y .- x[n, i]) .* (y .- x[n, i] .- 8.63)/(2.56^2) +
                   0.405*pdf.(Normal(15.24, 5.39), y .- x[n, i]) .* (y .- x[n, i] .- 15.24)/(5.39^2);
           drift[i] = mean(gradient./muN) + alpha*(x[n, i] .- m0)/sigma0^2;
       end
       # update locations
       x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ dt * abs.(drift)) .+  sqrt(2*alpha*dt)*randn(N, 1);
   end
   return x
end

#= WGF for deconvolution with Normal error
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'sigU' parameter for error distribution
=#
function wgf_DKDE_tamed(N, dt, Niter, alpha, x0, m0, sigma0, muSample, M, sigU)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # get samples from μ(y)
        y = sample(muSample, M, replace = false);
        # Compute denominator
        muN = zeros(M, 1);
        for j=1:M
            muN[j] =  mean(pdf.(Normal.(x[n, :], sigU), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Normal.(0, sigU), y .- x[n, i]) .* (y .- x[n, i])/sigU^2;
            drift[i] = mean(gradient./muN) + alpha*(x[n, i] .- m0)/sigma0^2;
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ dt * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

#= WGF for CT scan reconstruction
OUTPUTS
1 - particle locations (2D)
2 - value of functional E
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior'
'M' number of samples from h(y) to be drawn at each iteration
'sinogram' empirical distribution of data
'phi_angle' angle for projection data
'xi' depth of projection data
'sigma' standard deviation for Normal describing alignment
=#
function wgf_ct_tamed(N, dt, Niter, alpha, x0, m0, sigma0, M, sinogram, phi_angle, xi, sigma)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # intial distribution
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];
    # value of functional
    E = zeros(1, Niter);

    for n=1:(Niter-1)
        # log-likelihood
        # loglik = zeros(size(sinogram));
        # for i=1:length(phi_angle)
        #     for j=1:length(xi)
        #         loglik[i, j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(phi_angle[i]) .+
        #         x2[n, :] * sin(phi_angle[i]) .- xi[j]));
        #        end
        #     end
        # loglik = -log.(loglik);
        # kl = (phi_angle[2] - phi_angle[1])*(xi[2]-xi[1])*sum(loglik);
        # # prior
        # prior = pdf(MvNormal(m0, Diagonal(sigma0)), [x1[n, :] x2[n, :]]');
        # pihat = ct_kde([x1[n, :] x2[n, :]], [x1[n, :] x2[n, :]]);
        # kl_prior = mean(log.(pihat[:]./prior));
        # E[n] = kl+alpha*kl_prior;

        # get sample from μ(y)
        y = histogram2D_sampler(sinogram, xi, phi_angle, M);

        # Compute denominator
        muN = zeros(M, 1);
        for j=1:M
            muN[j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(y[j, 2]) .+
                    x2[n, :] * sin(y[j, 2]) .- y[j, 1]));
        end

        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x1[n, i] * cos.(y[:, 2]) .+
                    x2[n, i] * sin.(y[:, 2]) .- y[:, 1]) .*
                    (x1[n, i] * cos.(y[:, 2]) .+
                    x2[n, i] * sin.(y[:, 2]) .- y[:, 1])/sigma^2;
            gradientX1 = prec .* cos.(y[:, 2]);
            gradientX2 = prec .* sin.(y[:, 2]);
            # keep only finite elements
            g1h = gradientX1./muN;
            g2h = gradientX2./muN;
            driftX1[i] = mean(g1h) + alpha*(x1[n, i] .- m0[1])/sigma0[1]^2;
            driftX2[i] = mean(g2h) + alpha*(x2[n, i] .- m0[2])/sigma0[2]^2;
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ dt * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ dt * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    # loglik = zeros(size(sinogram));
    # for i=1:length(phi_angle)
    #     for j=1:length(xi)
    #         loglik[i, j] = mean(pdf.(Normal.(0, sigma), x1[Niter, :] * cos(phi_angle[i]) .+
    #         x2[Niter, :] * sin(phi_angle[i]) .- xi[j]));
    #        end
    #     end
    # loglik = -log.(loglik);
    # kl = (phi_angle[2] - phi_angle[1])*(xi[2]-xi[1])*sum(loglik);
    # # prior
    # prior = pdf(MvNormal(m0, Diagonal(sigma0)), [x1[Niter, :] x2[Niter, :]]');
    # pihat = ct_kde([x1[Niter, :] x2[Niter, :]], [x1[Niter, :] x2[Niter, :]]);
    # kl_prior = mean(log.(pihat[:]./prior));
    # E[Niter] = kl+alpha*kl_prior;
    return x1, x2, E
end

#= Kernel density estimatior for CT reconstructions
OUTPUTS
1 - KDE evaluated at KDEeval
INPUTS
'piSample' sample from π (Nx2 matrix)
'KDEeval' evaluation points (2 column matrix)
=#
function ct_kde(piSample, KDEeval)
    N = size(piSample, 1);
    # Silverman's plug in bandwidth
    bw1 = 1.06*Statistics.std(piSample[:, 1])*N^(-1/5);
    bw2 = 1.06*Statistics.std(piSample[:, 2])*N^(-1/5);

    KDEdensity = zeros(1, size(KDEeval, 1));
    for i = 1:size(KDEeval, 1)
        KDEdensity[i] = mean(pdf(MvNormal(KDEeval[i, :], diagm([bw1^2; bw2^2])), piSample'))/(bw1*bw2);
    end
    return KDEdensity;
end

#= WGF for cross validation of CT scan reconstruction
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior'
'muSample' sample from sinogram
'sigma' standard deviation for Normal describing alignment
=#
function wgf_ct_tamed_cv(N, dt, Niter, alpha, x0, m0, sigma0, muSample, sigma)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # intial distribution
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];

    for n=1:(Niter-1)
        # Compute denominator
        muN = zeros(M, 1);
        for j=1:M
            muN[j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(muSample[j, 2]) .+
                    x2[n, :] * sin(muSample[j, 2]) .- muSample[j, 1]));
        end

        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x1[n, i] * cos.(muSample[:, 2]) .+
                    x2[n, i] * sin.(muSample[:, 2]) .- y[:, 1]) .*
                    (x1[n, i] * cos.(muSample[:, 2]) .+
                    x2[n, i] * sin.(muSample[:, 2]) .- muSample[:, 1])/sigma^2;
            gradientX1 = prec .* cos.(muSample[:, 2]);
            gradientX2 = prec .* sin.(muSample[:, 2]);
            # keep only finite elements
            g1h = gradientX1./muN;
            g2h = gradientX2./muN;
            driftX1[i] = mean(g1h) + alpha*(x1[n, i] .- m0[1])/sigma0[1]^2;
            driftX2[i] = mean(g2h) + alpha*(x2[n, i] .- m0[2])/sigma0[2]^2;
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ dt * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ dt * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x1, x2
end

#= WGF for Gaussian mixture in d dimensions
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'muSample' sample from μ
'sigmaK' variance of mixing kernel
=#
function wgf_hd_mixture_tamed(N, dt, Niter, alpha, x0, m0, sigma0, muSample, sigmaK)
    # initial distribution
    x = copy(x0);
    # number of dimensions
    p = size(x0, 1);
    # number of samples from μ(y) to draw at each iteration
    M = min(N, size(muSample, 2));
    for n=1:(Niter-1)
        # get samples from μ(y)
        yIndex = sample(1:size(muSample, 2), M, replace = false);
        y = muSample[:, yIndex];
        # Compute denominator
        muN = zeros(M);
        for j=1:M
            muN[j] = mean(pdf(MvNormal(y[:, j], sigmaK^2*Matrix{Float64}(I, 2, 2)), x));
        end
        # gradient and drift
        drift = zeros(p, N);
        for i=1:N
            # precompute common quantities for gradient
            prec = pdf(MvNormal(x[:, i], sigmaK^2*Matrix{Float64}(I, 2, 2)), y);
            gradient = prec' .* (y .- x[:, i])/sigmaK^2;
            drift[:, i] =  mean(gradient./muN', dims = 2) .+ alpha*(x[:, i] .- m0)/sigma0^2;
        end
        # update locations
        drift_norm = sqrt.(sum(drift.^2, dims = 1));
        x = x .+ dt * drift./(1 .+ dt * drift_norm) .+ sqrt(2*alpha*dt)*randn(p, N);
    end
    return x
end
end
