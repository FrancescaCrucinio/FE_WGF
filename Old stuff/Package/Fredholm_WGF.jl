module Fredholm_WGF

using Distributions;
using Statistics;
import StatsBase: RealVector, RealMatrix

export FIE
export wgf_solve

# Fredholm integral equation object
# 'μ_sample' function to sample from μ
# 'K_eval' function to pointwise evaluate the kernel K
# 'ρ_support' support of the solution ρ
struct FIE
       μ_sample
       K_eval
       ρ_support
   end

function wgf_solve(fie::FIE, α::Float64, gradient::Function, Nparticles::Int=1000,
    dt::Float64=1e-03, Niter::Int=100, ρ0::RealMatrix=randn(size(fie.ρ_support, 1), Nparticles), M::Int=Nparticles)

    # input checks
    if α<=0
        error("the regularisation parameter alpha should be positive!")
    end
    if size(ρ0, 1) != size(fie.ρ_support, 1)
        error("initial distribution must have $size(fie.ρ_support, 1) dimensions!")
    end
    if size(ρ0, 2) != Nparticles
        error("initial distribution must give $Nparticles positions!")
    end
    if dt<1e-03
        warn("Finer time discretisations should be preferred")
    end
    # dimension of solution
    D = size(fie.ρ_support, 1);
    # initial distribution
    x = ρ0;
    # run WGF
    for n=1:(Niter-1)
        # get samples from μ
        y = fie.μ_sample(M);
        # Compute ρK
        ρK = zeros(M, 1);
        Threads.@threads for j=1:M
            ρK[j] = mean(fie.K_eval.(x, y[j]));
        end
        # gradient and drift
        drift = zeros(1, Nparticles);
        Threads.@threads for i=1:Nparticles
            gradient_eval = gradient.(x[i], y);
            drift[i] = mean(gradient_eval./ρK);
        end
        # update locations
        x = x .+ drift*dt .+ sqrt(2*α*dt)*randn(D, Nparticles);
    end
    return x
end
end
