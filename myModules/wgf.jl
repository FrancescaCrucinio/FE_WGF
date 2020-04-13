module wgf;

using Distributions;
using Statistics;
using samplers;

export wgf_AT_approximated
export wgf_gaussian_mixture
export wgf_deblurring
export wgf_pet
export wgf_mvnormal

#=
 WGF for analytically tractable example (approximated drift)
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_AT_approximated(N, Niter, lambda, x0, M)
    # time step
    dt = 1/Niter;
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
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, drift
end

# function exact_minimizer(sigmaG, sigmaH, lambda)
#     variance  = (sigmaH - sigmaG .+ 2*lambda*sigmaG +
#                 sqrt.(sigmaG^2 + sigmaH^2 .- 2*sigmaG*sigmaH.*(1 .- 2*lambda)))./
#                 (2*(1 .- lambda));
#     return variance
# end

#=
 WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'T' final time
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_gaussian_mixture(N, dt, T, lambda, x0, M)
    # number of iterations
    Niter = trunc(Int, T/dt);
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
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, drift
end

#=
 WGF for motion deblurring (constant speed motion)
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'I' data image
'M' number of samples from h(y) to be drawn at each iteration
'sigma' standard deviation for Normal approximating Dirac delta
'v' velocity of motion
'a', 'b' parameters of Beta approximating Uniform
=#
# function wgf_deblurring(N, Niter, lambda, I, M, sigma, v, a, b)
#     # normalize velocity
#     v = v/300;
#     # time step
#     dt = 1/Niter;
#     # initialise two matrices x, y storing the particles
#     x = zeros(Niter, N);
#     y = zeros(Niter, N);
#     # sample random particles for x in [-1, 1] for time step n = 1
#     x[1, :] = 2 * rand(1, N) .- 1;
#     # sample random particles for y in [-0.5, 0.5] for time step n = 1
#     y[1, :] = rand(1, N) .- 0.5;
#     # get samples from h(y)
#     pixels = size(I);
#     # x is in [-1, 1]
#     evalX = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
#     # y is in [-0.5, 0.5]
#     evalY = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
#     # get sample from (y)
#     hSample = histogram2D_sampler(I, evalX, evalY, M);
#     for n=1:(Niter-1)
#         # Compute h^N_{n}
#         hN = zeros(M, 1);
#         for j=1:M
#             # hN[j] = mean(pdf.(Normal.(0, sigma), hSample[j, 1] .- y[n, :]) .*
#             #         (x[n, :] .- hSample[j, 2] .<= b/2 &
#             #         x[n, :] .- hSample[j, 2] .>= -b/2)./b));
#             hN[j] = mean(pdf.(Normal.(0, sigma), hSample[j, 2] .- y[n, :]) .*
#                     pdf.(Beta(a, b), (x[n, :] .- hSample[j, 1] .+ v/2)./v) /v
#                  );
#         end
#         # gradient and drift
#         driftX = zeros(N, 1);
#         driftY = zeros(N, 1);
#         for i=1:N
#             # precompute normal and beta for gradient and drift
#             prec = pdf.(Normal.(0, sigma), hSample[:, 2] .- y[n, i]);
#             gradientX = - prec .* (a + b -2) * (a + b - 1) / v^3 .*
#                         pdf.(Beta(a-1, b-1), (x[n, i] .- hSample[:, 1] .+ v/2)./v);
#             gradientY = prec .* (hSample[:, 2] .- y[n, i])/(sigma^2) .*
#                         pdf.(Beta(a, b), (x[n, i] .- hSample[:, 1] .+ v/2)./v) /v;
#             driftX[i] = mean(gradientX./hN);
#             driftY[i] = mean(gradientY./hN);
#         end
#         # update locations
#         x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
#         y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
#     end
#     return x, y
# end

#=
 WGF for positron emission tomography
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'noisyI' data image (Radon transform)
'M' number of samples from h(y) to be drawn at each iteration
'phi' degrees at which projections are taken
'xi' offset of projections
'sigma' standard deviation for Normal describing alignment
=#
function wgf_pet(N, Niter, lambda, noisyI, M, phi, xi, sigma)
    # time step
    dt = 1/Niter;
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
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, y
end

#=
 WGF for multivariate normal
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
'mu' mean of data Distribution
'sigmaH' covariance matrix of data distribution
'sigmaG' covariance matrix of mixing kernel
=#
function wgf_mvnormal(N, Niter, lambda, x0, M, mu, sigmaH, sigmaG)
    # time step
    dt = 1/Niter;
    # initialise two matrices x, y storing the particles
    x = zeros(Niter, N);
    y = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0[1, :];
    y[1, :] = x0[2, :];

    for n=1:(Niter-1)
        # get samples from h(y)
        hSample = rand(MvNormal(mu, sigmaH), M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            # define Gaussian pdf
            phi(t) = pdf(MvNormal(hSample[:, j], sigmaG), t);
            # apply it to c, y
            hN[j] = mean(mapslices(phi, [x[n, :] y[n, :]], dims = 2));
        end
        # gradient and drift
        driftX = zeros(N, 1);
        driftY = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            # define Gaussian pdf
            psi(t) = pdf(MvNormal([x[n, i]; y[n, i]], sigmaG), t);
            prec = mapslices(psi, hSample', dims = 2) .*
                ((hSample[2, :] .- y[n, i])/sqrt(sigmaG[2, 2]) -
                (hSample[1, :] .- x[n, i])/sqrt(sigmaG[1, 1]));
            prec = sigmaG[2, 2] * sigmaG[1, 1] *
                prec./(sigmaG[2, 2] * sigmaG[1, 1] - sigmaG[1, 2]^2);
            gradientX = -prec./sqrt(sigmaG[1, 1]);
            gradientY = prec./sqrt(sigmaG[2, 2]);
            driftX[i] = mean(gradientX./hN);
            driftY[i] = mean(gradientY./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, y
end
end
