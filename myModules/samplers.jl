module samplers

using Statistics;
using StatsBase;
using IterTools;
using QuadGK;

export Ysample_gaussian_mixture
export histogram2D_sampler
export walker_sampler
export rejection_sampling_2D_analytic

# Sample y from the gaussian mixture model
# OUTPUTS
# 1 - sample from h
# INPUTS
# 'M' number of samples
function Ysample_gaussian_mixture(M)
    #Mixture
    yl = rand(M,1) .> 1/3;
    # mean
    ym = 0.3 .+ 0.2 * yl;
    # variance
    yv = zeros(size(ym));
    for i=1:length(yl)
        if(yl[i])
    	  yv[i] = 0.043^2 + 0.045^2;
        else
    	  yv[i] = 0.015^2 + 0.045^2;
        end
    end
    # traslated standard normal
    y = ym .+ sqrt.(yv).*randn(M,1);
    y = y[:];
end

# Sample from 2D histogram
# OUTPUTS
# 1 - sample of size M
# INPUTS
# 'Image' 2D histogram (or image)
# 'x' values on x coordinate
# 'y' values on y coordinate
# 'M' number of samples
function histogram2D_sampler(Image, x, y, M)
    # dimensions of matrix
    r = size(Image, 1);
    c = size(Image, 2);
    # cartesian product of indices
    indices = collect(Iterators.product(1:r, 1:c));
    # vector of weights
    w = Image[:];
    # walker sampler
    samples_indices = indices[walker_sampler(w, M)];
    # flatten array
    samples_indices = collect(Iterators.flatten(samples_indices));
    # odd entries are rows and even entries are columns
    samples = [x[samples_indices[2:2:end]] y[samples_indices[1:2:end]]];

    return samples
end
# Walker's sampler for discrete probabilities
# OUTPUTS
# 1 - sampled indices of w
# INPUTS
# 'w' vector of weights (not normalised)
# 'M' number of samples
function walker_sampler(w, M)
    # lenght of w
    n = length(w);
    # probability and alias walker_matrix
    pm, am = walker_matrix(w);
    out = zeros(M, 1);
    unif = rand(M, 1);
    j = rand(1:n, M);
    out = ifelse.(unif .< pm[j], j, am[j]);
    out = trunc.(Int, out);
    return out
end
# Probability and aliasing matrix for Walker's sampler
# OUTPUTS
# 1 - probability matrix
# 2 - aliasing matrix
# INPUTS
# 'w' vector of weights (not normalised)
function walker_matrix(w)
    # lenght of w
    n = length(w);
    # normalise and multiply by length to get probability table
    w = n * w./sum(w);
    # overfull and underfull group
    overfull = findall(w .> 1);
    underfull = findall(w .< 1);
    alias_m = -ones(n);
    # make entries exactly full
    while (!isempty(overfull) & !isempty(underfull))
        j = pop!(underfull);
        i = overfull[end];
        alias_m[j] = i;
        w[i] = w[i] - 1 + w[j];
        if (w[i] < 1)
                append!(underfull, i);
                pop!(overfull);
        end
    end
    return w, alias_m
end
end

# # Rejection sampling for 2D analytic example
# # OUTPUTS
# # 1 - sample from h
# # INPUTS
# # 'M' number of samples
# function rejection_sampling_2D_analytic(M)
#     # integral sine function
#     Si(x) = quadgk(t -> sin(t)/t, 0, x)[1];
#     # h
#     h(y) = ((y[1].^2 .+ y[2].^2)*Si(1) .- 2*(1-cos(1))*(y[1] .+ y[2]) + 2*sin(1) - 2*cos(1))/
#         (Si(1)*2/3-2+2*sin(1));
#     # constant for acceptance probability
#     C = 2.1;
#
#     hSample = zeros(M, 2);
#     j = 0;
#     while(j < M)
#         # sample uniform in [0, pi]^2
#         uSample = rand(2, 1);
#         # accept/reject
#         if (C * rand(1)[1] < h(uSample))
#             hSample[j+1, :] =  uSample;
#             j = j + 1;
#         end
#     end
#     return hSample
# end
