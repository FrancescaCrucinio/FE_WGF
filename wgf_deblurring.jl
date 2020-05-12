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
