module wgf;

using Distributions;
using Statistics;
using LinearAlgebra;

using samplers;

export wgf_AT_tamed
export wgf_gaussian_mixture_tamed
export wgf_pet_tamed
export AT_exact_minimiser
export wgf_sucrase_tamed
export wgf_DKDE_tamed
export wgf_flu_tamed
export wgf_mvnormal_tamed

#= WGF for analytically tractable example
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_AT_tamed(N, dt, Niter, alpha, x0, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # initialise a matrix drift storing the drift
    drift = zeros(Niter-1, N);

    for n=1:(Niter-1)
        # get samples from h(y)
        y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[n, i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift[n, :]./(1 .+ Niter^(-a) * abs.(drift[n, :])) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x, drift
end

#= WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_gaussian_mixture_tamed(N, dt, Niter, alpha, x0, muSample, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+  dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

#= WGF for positron emission tomography
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'muSample' sample from noisy image μ(y)
'M' number of samples from h(y) to be drawn at each iteration
'sigma' standard deviation for Normal describing alignment
'a' parameter for tamed Euler scheme
=#
function wgf_pet_tamed(N, dt, Niter, alpha, muSample, M, sigma, a)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # intial distribution
    x0 = rand(MvNormal([0, 0], 0.1*Diagonal(ones(2))), N);
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];
    for n=1:(Niter-1)
        # get sample from μ(y)
        muIndex = sample(1:size(muSample, 1), M, replace = true);
        y = muSample[muIndex, :];
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(y[j, 1]) .+
                    x2[n, :] * sin(y[j, 1]) .- y[j, 2])
                    );
        end
        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2]) .*
                    (x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2])/sigma^2;
            gradientX1 = prec .* cos.(y[:, 1]);
            gradientX2 = prec .* sin.(y[:, 1]);
            # keep only finite elements
            g1h = gradientX1./hN;
            g2h = gradientX2./hN;
            g1h[(!).(isfinite.(g1h))] .= 0;
            g2h[(!).(isfinite.(g2h))] .= 0;
            driftX1[i] = mean(g1h);
            driftX2[i] = mean(g2h);
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x1, x2
end

#= Exact minimiser for analytically tractable example
OUTPUTS
1 - variance
2 - KL divergence
INPUTS
'sigmaK' variance of kernel K
'sigmaMu' variance of data function μ
'alpha' regularisation parameter
=#
function AT_exact_minimiser(sigmaK, sigmaMu, alpha)
    variance  = (sigmaMu - sigmaK .+ 2*alpha*sigmaK .+
                sqrt.(sigmaK^2 + sigmaMu^2 .- 2*sigmaK*sigmaMu*(1 .- 2*alpha)))./
                (2*(1 .- alpha));
    KL = 0.5*log.((sigmaK .+ variance)/sigmaMu) .+ 0.5*sigmaMu./(sigmaK .+ variance) .- 0.5 .-
        0.5*alpha .* (1 .+ log.(2*pi*variance));
    return variance, KL
end

#= WGF for deconvolution with real data (sucrase example)
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
'sigU' parameter for error distribution
=#
function wgf_sucrase_tamed(N, dt, Niter, alpha, x0, muSample, M, a, sigU)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # get samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            # hN[j] = mean(pdf.(Normal.(x[n, :], sigU), y[j]));
            hN[j] = mean(pdf.(Laplace.(x[n, :], sigU), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            # gradient = pdf.(Normal.(x[n, i], sigU), y) .* (y .- x[n, i])/(sigU^2);
            gradient = pdf.(Laplace.(x[n, i], sigU), y) .* (-sign.(x[n, i] .- y)/sigU);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

#= WGF for deconvolution with simulated data and Laplace error
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
'sigU' parameter for error distribution
=#
function wgf_DKDE_tamed(N, dt, Niter, alpha, x0, muSample, M, a, sigU)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # get samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Laplace.(x[n, :], sigU), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Laplace.(x[n, i], sigU), y) .* (-sign.(x[n, i] .- y)/sigU);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end
#= WGF for Spanish flu data
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_flu_tamed(N, dt, Niter, alpha, x0, muSample, M, a)
   # initialise a matrix x storing the particles
   x = zeros(Niter, N);
   # initial distribution is given as input:
   x[1, :] = x0;

   for n=1:(Niter-1)
       # get samples from h(y)
       y = sample(muSample, M, replace = true);
       # Compute h^N_{n}
       hN = zeros(M, 1);
       for j=1:M
           hN[j] = mean(0.595*pdf.(Normal(8.63, 2.56), y[j] .- x[n, :]) +
                   0.405*pdf.(Normal(15.24, 5.39), y[j] .- x[n, :]))
       end

       # gradient and drift
       drift = zeros(N, 1);
       for i=1:N
           gradient = 0.595*pdf.(Normal(8.63, 2.56), y .- x[n, i]) .* (y .- x[n, i] .- 8.63)/(2.56^2) +
                   0.405*pdf.(Normal(15.24, 5.39), y .- x[n, i]) .* (y .- x[n, i] .- 15.24)/(5.39^2);
           drift[i] = mean(gradient./hN);
       end
       # update locations
       x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
   end
   return x
end

#=
 WGF for gaussian mixture 2D
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of time steps
'alpha' regularisation parameter
'x0' user selected initial distribution
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_mvnormal_tamed(N, dt, Niter, alpha, x0, muSample, M, a)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # initial distribution is given as input:
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];
    # covariance for K
    sigmaK = [0.1 0; 0 1];

    for n=1:(Niter-1)
        # get sample from μ(y)
        muIndex = sample(1:size(muSample, 2), M, replace = true);
        y = muSample[:, muIndex];
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf(MvNormal(y[:, j], sigmaK), transpose([x1[n, :] x2[n, :]])));
        end
        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = pdf(MvNormal([x1[n, i]; x2[n, i]], sigmaK), y);
            gradientX1 = prec .* (y[1, :] .- x1[n, i])/sigmaK[1, 1];
            gradientX2 = prec .* (y[2, :] .- x2[n, i])/sigmaK[2, 2];
            driftX1[i] = mean(gradientX1./hN);
            driftX2[i] = mean(gradientX2./hN);
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x1, x2
end
end




# #=
#  WGF for analytically tractable example
# OUTPUTS
# 1 - particle locations
# 2 - drift evolution
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of iterations
# 'lambda' regularisation parameter
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function wgf_AT(N, dt, Niter, lambda, x0, M)
#     # initialise a matrix x storing the particles
#     x = zeros(Niter, N);
#     # initial distribution is given as input:
#     x[1, :] = x0;
#     # initialise a matrix drift storing the drift
#     drift = zeros(Niter-1, N);
#
#     for n=1:(Niter-1)
#         # get samples from h(y)
#         y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
#         # Compute h^N_{n}
#         hN = zeros(M, 1);
#         for j=1:M
#             hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
#         end
#         # gradient and drift
#         for i=1:N
#             gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
#             drift[n, i] = mean(gradient./hN);
#         end
#         # update locations
#         x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#     end
#     return x, drift
# end
#
# WGF for multivariate normal
# OUTPUTS
# 1 - particle locations (2D)
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of time steps
# 'lambda' regularisation parameter
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function wgf_2D_analytic(N, dt, Niter, lambda, x0, M)
#    # kernel g
#    g(x, y) = ((y[1] .- x[1]).^2 .+ (y[2] .- x[2]).^2)./
#        (x[1].^2 .+ x[2].^2 .- x[1] .- x[2] .+ 2/3);
#    # initialise two matrices x, y storing the particles
#    x = zeros(Niter, N);
#    y = zeros(Niter, N);
#    # initial distribution is given as input:
#    x[1, :] = x0[1, :];
#    y[1, :] = x0[2, :];
#
#    for n=1:(Niter-1)
#        # get samples from h(y)
#        hSample = rejection_sampling_2D_analytic(M);
#        # Compute h^N_{n}
#        hN = zeros(M, 1);
#        for j=1:M
#            # define Gaussian pdf
#            phi(t) = g(t, hSample[j, :]);
#            # apply it to c, y
#            hN[j] = mean(mapslices(phi, [x[n, :] y[n, :]], dims = 1));
#        end
#        # gradient and drift
#        driftX = zeros(N, 1);
#        driftY = zeros(N, 1);
#        for i=1:N
#            # precompute common quantities for gradient
#            # denominators
#            prec = x[n, i]^2 + y[n,i]^2 - x[n, i] - y[n,i] + 2/3;
#            gradientX = -2*(hSample[:, 1] .- x[n, i])/prec -
#            (2*x[n, i] - 1)*((hSample[:, 1] .- x[n, i]).^2 + (hSample[:, 2] .- y[n, i]).^2)/prec^2;
#            gradientY = -2*(hSample[:, 2] .- y[n, i])/prec -
#            (2*y[n, i] - 1)*((hSample[:, 1] .- x[n, i]).^2 + (hSample[:, 2] .- y[n, i]).^2)/prec^2;
#            driftX[i] = mean(gradientX./hN);
#            driftY[i] = mean(gradientY./hN);
#        end
#        # update locations
#        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#    end
#    return x, y
# end
#
# #= For entropy computation - removes non finite entries =#
# function remove_non_finite(x)
#       return isfinite(x) ? x : zero(x)
# end
#
# #= Find α giving a target entropy for WGF
# OUTPUTS
# 1 - alpha
# INPUTS
# 'target_entropy' target entropy for the solution
# 'interval' domain of α
# 'threshold' stopping rule
# 'dt' discretisation step
# 'Niter' number of time steps
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function AT_alpha_WGF(target_entropy, interval, threshold, dt, Niter, Nparticles, initial_distribution, M)
#    # values at which evaluate KDE
#    KDEx = range(0, stop = 1, length = 1000);
#    # upper and lower bound for α
#    liminf = interval[1];
#    limsup = interval[2];
#
#    delta_entropy = Inf;
#    alpha = (limsup + liminf)/2;
#    Nrep = 10;
#    j=1;
#    while (abs(delta_entropy)>threshold && j<=50)
#        actual_entropy  = zeros(Nrep, 1);
#        Threads.@threads for i=1:Nrep
#            if (initial_distribution == "delta")
#                x0 = rand(1)*ones(1, Nparticles);
#            else
#                x0 = rand(1, Nparticles);
#            end
#            ### WGF
#            xWGF, _ =  wgf_AT(Nparticles, dt, Niter, alpha, x0, M);
#            # KDE
#            # optimal bandwidth Gaussian
#            KDEyWGF =  KernelEstimator.kerneldensity(xWGF[end,:], xeval=KDEx, h=bwnormal(xWGF[end,:]));
#            actual_entropy[i] = -mean(remove_non_finite.(KDEyWGF .* log.(KDEyWGF)));
#        end
#        actual_entropy = mean(actual_entropy);
#        delta_entropy = actual_entropy - target_entropy;
#        if (delta_entropy > 0)
#            limsup = (limsup + liminf)/2;
#        else
#            liminf = (limsup + liminf)/2;
#        end
#        alpha = (limsup + liminf)/2;
#        println("$j")
#        println("$limsup , $liminf")
#        println("$actual_entropy")
#        println("$delta_entropy")
#        j=j+1;
#    end
#    return limsup, liminf
# end
#
# #=
# WGF for motion deblurring (constant speed motion)
# OUTPUTS
# 1 - particle locations (2D)
# INPUTS
# 'N' number of particles
# 'Niter' number of time steps
# 'lambda' regularisation parameter
# 'I' data image
# 'M' number of samples from h(y) to be drawn at each iteration
# 'sigma' standard deviation for Normal approximating Dirac delta
# 'a' acceleration of motion
# =#
# function wgf_turbolence(N, Niter, dt, lambda, I, M, beta, R)
#    # normalize acceleration
#    R = R/300;
#    # initialise two matrices x, y storing the particles
#    x = zeros(Niter, N);
#    y = zeros(Niter, N);
#    # sample random particles for x in [-1, 1] for time step n = 1
#    x[1, :] = randn(1, N)/3;
#    # sample random particles for y in [-0.5, 0.5] for time step n = 1
#    y[1, :] = randn(1, N)/6;
#    # get samples from h(y)
#    pixels = size(I);
#    # x is in [-1, 1]
#    evalX = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
#    # y is in [-0.5, 0.5]
#    evalY = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
#    for n=1:(Niter-1)
#        # get sample from (y)
#        hSample = histogram2D_sampler(I, evalX, evalY, M);
#        # Compute h^N_{n}
#        hN = zeros(M, 1);
#        for j=1:M
#            hN[j] = mean((1 .+ ((hSample[j, 1] .- x[n, :]).^2 .+
#                (hSample[j, 2] .- y[n, :]).^2)/R^2).^(-beta));
#        end
#        # gradient and drift
#        driftX = zeros(N, 1);
#        driftY = zeros(N, 1);
#        for i=1:N
#            # precompute normal and beta for gradient and drift
#            prec = (2*beta/R^4) * (1 .+ ((hSample[:, 1] .- x[n, i]).^2 .+
#                (hSample[:, 2] .- y[n, i]).^2)/R^2).^(-beta-1);
#            gradientX = prec .* (hSample[:, 1] .- x[n, i]);
#            gradientY = prec .* (hSample[:, 2] .- y[n, i]);
#            driftX[i] = mean(gradientX./hN);
#            driftY[i] = mean(gradientY./hN);
#        end
#        # update locations
#        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#    end
#    return x, y
# end
#
# #=
# WGF for analytically tractable example
# OUTPUTS
# 1 - particle locations
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of iterations
# 'lambda' regularisation parameter
# 'x0' user selected initial distribution
# 'hSample' sample from h(y)
# 'M' number of samples from h(y) to be drawn at each iteration
# 'a' parameter for tamed Euler scheme
# =#
# function wgf_C19_tamed(N, dt, Niter, lambda, x0, hSample, M, a)
#    # initialise a matrix x storing the particles
#    x = zeros(Niter, N);
#    # initial distribution is given as input:
#    x[1, :] = x0;
#
#    alpha = 4.9^2/3.3^2;
#    beta = 4.9/3.3.^2;
#    sigma = sqrt(2log(5.5/5.2));
#    mu = log(5.2);
#
#    for n=1:(Niter-1)
#        # get samples from h(y)
#        y = sample(hSample, M, replace = true);
#        # Compute h^N_{n}
#        hN = zeros(M, 1);
#        for j=1:M
#            hN[j] = mean(C19_K(x[n, :] .- y[j], alpha, beta, mu, sigma));
#        end
#        # gradient and drift
#        drift = zeros(N, 1);
#        for i=1:N
#            gradient = -beta^2 .* C19_K(x[n, i] .- y, alpha-1, beta, mu, sigma);
#            drift[i] = mean(gradient./hN);
#        end
#        # update locations
#        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*lambda*dt)*randn(N, 1);
#    end
#    return x
# end
#
# #=
# WGF for analytically tractable example
# OUTPUTS
# 1 - particle locations
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of iterations
# 'lambda' regularisation parameter
# 'x0' user selected initial distribution
# 'hSample' sample from h(y)
# 'M' number of samples from h(y) to be drawn at each iteration
# 'a' parameter for tamed Euler scheme
# =#
# function wgf_HIV_tamed(N, dt, Niter, alpha, x0, hSample, M, a)
#    # initialise a matrix x storing the particles
#    x = zeros(Niter, N);
#    # initial distribution is given as input:
#    x[1, :] = x0;
#
#    kappa = 2.516;
#    lambda = 8/ log(2)^(1/kappa);
#
#    for n=1:(Niter-1)
#        # get samples from h(y)
#        y = sample(hSample, M, replace = true);
#        # Compute h^N_{n}
#        hN = zeros(M, 1);
#        for j=1:M
#            hN[j] = mean(pdf.(Weibull(kappa, lambda), y[j] .- x[n, :] .+ 1));
#        end
#        # gradient and drift
#        drift = zeros(N, 1);
#        for i=1:N
#            gradient = zeros(M, 1);
#            positive = (y .- x[n, i]).>0;
#            gradient[positive] = pdf.(Weibull(kappa, lambda),  y[positive] .- x[n, i]  .+ 1) .*
#                ((kappa-1)*lambda./(y[positive] .- x[n, i] .+ 1) - kappa*(y[positive] .- x[n, i] .+ 1).^(kappa-1)./lambda^kappa);
#            ratio = gradient./hN;
#            ratio[isnan.(ratio)] .= 0;
#            drift[i] = mean(ratio);
#        end
#        # update locations
#        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+ sqrt(2*alpha*dt)*randn(N, 1);
#    end
#    return x
# end
#

# #=
#  WGF for multivariate normal
# OUTPUTS
# 1 - particle locations (2D)
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of time steps
# 'lambda' regularisation parameter
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# 'mu' mean of data Distribution
# 'sigmaH' covariance matrix of data distribution
# 'sigmaG' covariance matrix of mixing kernel
# =#
# function wgf_mvnormal(N, dt, Niter, lambda, x0, M, mu, sigmaH, sigmaG)
#     # initialise two matrices x, y storing the particles
#     x = zeros(Niter, N);
#     y = zeros(Niter, N);
#     # initial distribution is given as input:
#     x[1, :] = x0[1, :];
#     y[1, :] = x0[2, :];
#     # correlation for g
#     rhoG =  sigmaG[1, 2]/sqrt(sigmaG[1, 1] * sigmaG[2, 2]);
#
#     for n=1:(Niter-1)
#         # get samples from h(y)
#         hSample = rand(MvNormal(mu, sigmaH), M);
#         # Compute h^N_{n}
#         hN = zeros(M, 1);
#         Threads.@threads for j=1:M
#             hN[j] = mean(pdf(MvNormal(hSample[:, j], sigmaG), transpose([x[n, :] y[n, :]])));
#         end
#         # gradient and drift
#         driftX = zeros(N, 1);
#         driftY = zeros(N, 1);
#         Threads.@threads for i=1:N
#             # precompute common quantities for gradient
#             prec =pdf(MvNormal([x[n, i]; y[n, i]], sigmaG), hSample)/(1 - rhoG^2);
#             gradientX = prec .* ((hSample[1, :] .- x[n, i])/sigmaG[1, 1] -
#                 rhoG*(hSample[2, :] .- y[n, i])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
#             gradientY = prec .* ((hSample[2, :] .- y[n, i])/sigmaG[2, 2] -
#                 rhoG*(hSample[1, :] .- x[n, i])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
#             driftX[i] = mean(gradientX./hN);
#             driftY[i] = mean(gradientY./hN);
#         end
#         # update locations
#         x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#         y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#     end
#     return x, y
# end

# #=
#  WGF for gaussian mixture
# OUTPUTS
# 1 - particle locations
# 2 - drift evolution
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of iterations
# 'alpha' regularisation parameter
# 'x0' user selected initial distribution
# 'hSample' sample from h(y)
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function wgf_gaussian_mixture(N, dt, Niter, alpha, x0, hSample, M)
#     # initialise a matrix x storing the particles
#     x = zeros(Niter, N);
#     # initial distribution is given as input:
#     x[1, :] = x0;
#
#     for n=1:(Niter-1)
#         # samples from h(y)
#         y = sample(hSample, M, replace=true);
#         # Compute h^N_{n}
#         hN = zeros(M, 1);
#         for j=1:M
#             hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
#         end
#         # gradient and drift
#         drift = zeros(N, 1);
#         for i=1:N
#             gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
#             drift[i] = mean(gradient./hN);
#         end
#         # update locations
#         x[n+1, :] = x[n, :] .+ drift*dt .+ sqrt(2*alpha*dt)*randn(N, 1);
#     end
#     return x
# end

#
# #=
#  WGF for positron emission tomography
# OUTPUTS
# 1 - particle locations (2D)
# INPUTS
# 'N' number of particles
# 'dt' discretisation step
# 'Niter' number of iterations
# 'lambda' regularisation parameter
# 'noisyI' data image (Radon transform)
# 'M' number of samples from h(y) to be drawn at each iteration
# 'phi' degrees at which projections are taken
# 'xi' offset of projections
# 'sigma' standard deviation for Normal describing alignment
# =#
# function wgf_pet(N, dt, Niter, lambda, noisyI, M, phi, xi, sigma)
#     # normalise xi
#     xi = xi./maximum(xi);
#     # initialise two matrices x, y storing the particles
#     x = zeros(Niter, N);
#     y = zeros(Niter, N);
#     # # sample random particles for x in [-1, 1] for time step n = 1
#     # x[1, :] = 2 * rand(1, N) .- 1;
#     # # sample random particles for y in [-1, 1] for time step n = 1
#     # y[1, :] = 2 * rand(1, N) .- 1;
#     x0 = rand(MvNormal([0, 0], Matrix{Float64}(I, 2, 2)), N);
#     x[1, :] = x0[1, :];
#     y[1, :] = x0[2, :];
#     for n=1:(Niter-1)
#         # get sample from (y)
#         hSample = histogram2D_sampler(noisyI, phi, xi, M);
#         # Compute h^N_{n}
#         hN = zeros(M, 1);
#         Threads.@threads for j=1:M
#             hN[j] = mean(pdf.(Normal.(0, sigma), x[n, :] * cos(hSample[j, 1]) .+
#                     y[n, :] * sin(hSample[j, 1]) .- hSample[j, 2])
#                     );
#         end
#         # gradient and drift
#         driftX = zeros(N, 1);
#         driftY = zeros(N, 1);
#         Threads.@threads for i=1:N
#             # precompute common quantities for gradient
#             prec = -pdf.(Normal.(0, sigma), x[n, i] * cos.(hSample[:, 1]) .+
#                     y[n, i] * sin.(hSample[:, 1]) .- hSample[:, 2]) .*
#                     (x[n, i] * cos.(hSample[:, 1]) .+
#                     y[n, i] * sin.(hSample[:, 1]) .- hSample[:, 2])/sigma^2;
#             gradientX = prec .* cos.(hSample[:, 1]);
#             gradientY = prec .* sin.(hSample[:, 1]);
#             driftX[i] = mean(gradientX./hN);
#             driftY[i] = mean(gradientY./hN);
#         end
#         # update locations
#         x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#         y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
#     end
#     return x, y
# end
