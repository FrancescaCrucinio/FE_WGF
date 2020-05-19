module wgf;

using Distributions;
using Statistics;
using LinearAlgebra;

using samplers;

export wgf_AT
export wgf_gaussian_mixture
export wgf_pet
export wgf_mvnormal
export AT_exact_minimiser
export wgf_2D_analytic

#=
 WGF for analytically tractable example
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_AT(N, dt, Niter, lambda, x0, M)
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
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, drift
end

#=
 WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_gaussian_mixture(N, dt, Niter, lambda, x0, M)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # initialise a matrix drift storing the drift
    drift = zeros(Niter-1, N);

    for n=1:(Niter-1)
        # get samples from h(y)
        y = Ysample_gaussian_mixture(M);
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
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, drift
end

#=
 WGF for positron emission tomography
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'noisyI' data image (Radon transform)
'M' number of samples from h(y) to be drawn at each iteration
'phi' degrees at which projections are taken
'xi' offset of projections
'sigma' standard deviation for Normal describing alignment
=#
function wgf_pet(N, dt, Niter, lambda, noisyI, M, phi, xi, sigma)
    # normalise xi
    xi = xi./maximum(xi);
    # initialise two matrices x, y storing the particles
    x = zeros(Niter, N);
    y = zeros(Niter, N);
    # sample random particles for x in [-1, 1] for time step n = 1
    x[1, :] = 2 * rand(1, N) .- 1;
    # sample random particles for y in [-1, 1] for time step n = 1
    y[1, :] = 2 * rand(1, N) .- 1;

    for n=1:(Niter-1)
        # get sample from (y)
        hSample = histogram2D_sampler(noisyI, phi, xi, M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sigma), x[n, :] * cos(hSample[j, 1]) .+
                    y[n, :] * sin(hSample[j, 1]) .- hSample[j, 2])
                    );
        end
        # gradient and drift
        driftX = zeros(N, 1);
        driftY = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x[n, i] * cos.(hSample[:, 1]) .+
                    y[n, i] * sin.(hSample[:, 1]) .- hSample[:, 2]) .*
                    (x[n, i] * cos.(hSample[:, 1]) .+
                    y[n, i] * sin.(hSample[:, 1]) .- hSample[:, 2])/sigma^2;
            gradientX = prec .* cos.(hSample[:, 1]);
            gradientY = prec .* sin.(hSample[:, 1]);
            driftX[i] = mean(gradientX./hN);
            driftY[i] = mean(gradientY./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, y
end

#=
 WGF for multivariate normal
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
'mu' mean of data Distribution
'sigmaH' covariance matrix of data distribution
'sigmaG' covariance matrix of mixing kernel
=#
function wgf_mvnormal(N, dt, Niter, lambda, x0, M, mu, sigmaH, sigmaG)
    # initialise two matrices x, y storing the particles
    x = zeros(Niter, N);
    y = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0[1, :];
    y[1, :] = x0[2, :];
    # correlation for g
    rhoG =  sigmaG[1, 2]/sqrt(sigmaG[1, 1] * sigmaG[2, 2]);

    for n=1:(Niter-1)
        # get samples from h(y)
        hSample = rand(MvNormal(mu, sigmaH), M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf(MvNormal(hSample[:, j], sigmaG), transpose([x[n, :] y[n, :]])));
        end
        # gradient and drift
        driftX = zeros(N, 1);
        driftY = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec =pdf(MvNormal([x[n, i]; y[n, i]], sigmaG), hSample)/(1 - rhoG^2);
            gradientX = prec .* ((hSample[1, :] .- x[n, i])/sigmaG[1, 1] -
                rhoG*(hSample[2, :] .- y[n, i])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
            gradientY = prec .* ((hSample[2, :] .- y[n, i])/sigmaG[2, 2] -
                rhoG*(hSample[1, :] .- x[n, i])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
            driftX[i] = mean(gradientX./hN);
            driftY[i] = mean(gradientY./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, y
end

#= Exact minimiser for analytically tractable example
OUTPUTS
1 - variance
2 - KL divergence
INPUTS
'sigmaG' variance of kernel g
'sigmaH' variance of data function h
'lambda' regularisation parameter
=#
function AT_exact_minimiser(sigmaG, sigmaH, lambda)
    variance  = (sigmaH - sigmaG .+ 2*lambda*sigmaG .+
                sqrt.(sigmaG^2 + sigmaH^2 .- 2*sigmaG*sigmaH*(1 .- 2*lambda)))./
                (2*(1 .- lambda));
    KL = 0.5*log.((sigmaG .+ variance)/sigmaH) .+ sigmaH./(sigmaG .+ variance) .- 0.5 .-
        0.5*lambda .* (1 .+ log.(2*pi*variance));
    return variance, KL
end

#=
 WGF for multivariate normal
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_2D_analytic(N, dt, Niter, lambda, x0, M)
    # kernel g
    g(x, y) = ((y[1] .- x[1]).^2 .+ (y[2] .- x[2]).^2)./
        (x[1].^2 .+ x[2].^2 .- x[1] .- x[2] .+ 2/3);
    # initialise two matrices x, y storing the particles
    x = zeros(Niter, N);
    y = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0[1, :];
    y[1, :] = x0[2, :];

    for n=1:(Niter-1)
        # get samples from h(y)
        hSample = rejection_sampling_2D_analytic(M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            # define Gaussian pdf
            phi(t) = g(t, hSample[j, :]);
            # apply it to c, y
            hN[j] = mean(mapslices(phi, [x[n, :] y[n, :]], dims = 1));
        end
        # gradient and drift
        driftX = zeros(N, 1);
        driftY = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            # denominators
            prec = x[n, i]^2 + y[n,i]^2 - x[n, i] - y[n,i] + 2/3;
            gradientX = -2*(hSample[:, 1] .- x[n, i])/prec -
            (2*x[n, i] - 1)*((hSample[:, 1] .- x[n, i]).^2 + (hSample[:, 2] .- y[n, i]).^2)/prec^2;
            gradientY = -2*(hSample[:, 2] .- y[n, i])/prec -
            (2*y[n, i] - 1)*((hSample[:, 1] .- x[n, i]).^2 + (hSample[:, 2] .- y[n, i]).^2)/prec^2;
            driftX[i] = mean(gradientX./hN);
            driftY[i] = mean(gradientY./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, y
end
end
