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
'a' acceleration of motion
=#
function wgf_deblurring(N, Niter, dt, lambda, I, M, sigma, a)
    # normalize acceleration
    a = a/300;
    # initialise two matrices x, y storing the particles
    x = zeros(Niter, N);
    y = zeros(Niter, N);
    # sample random particles for x in [-1, 1] for time step n = 1
    x[1, :] = 2 * rand(1, N) .- 1;
    # sample random particles for y in [-0.5, 0.5] for time step n = 1
    y[1, :] = rand(1, N) .- 0.5;
    # get samples from h(y)
    pixels = size(I);
    # x is in [-1, 1]
    evalX = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
    # y is in [-0.5, 0.5]
    evalY = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
    for n=1:(Niter-1)
        # get sample from (y)
        hSample = histogram2D_sampler(I, evalX, evalY, M);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sigma), hSample[j, 2] .- y[n, :]) .*
                    pdf.(Beta(0.5, 1), (hSample[j, 1] .- x[n, :])/a)
                 );
        end
        # gradient and drift
        driftX = zeros(N, 1);
        driftY = zeros(N, 1);
        for i=1:N
            # precompute normal and beta for gradient and drift
            prec = pdf.(Normal.(0, sigma), hSample[:, 2] .- y[n, i]);
            gradientX = prec .* pdf.(Beta(0.5, 1), (hSample[:, 1] .- x[n, i])/a) ./
                        (2*(hSample[:, 1] .- x[n, i]));
            gradientY = prec .* (hSample[:, 2] .- y[n, i])/(sigma^2) .*
                        pdf.(Beta(0.5, 1), (hSample[:, 1] .- x[n, i])/a);
            driftX[i] = mean(gradientX./hN);
            driftY[i] = mean(gradientY./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ driftX*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
        y[n+1, :] = y[n, :] .+ driftY*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, y
end
