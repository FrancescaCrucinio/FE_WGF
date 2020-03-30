module samplers

using Statistics;
using StatsBase;
using IterTools;

export Ysample_gaussian_mixture
export histogram2D_sampler

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
end

# Sample from 2D histogram
# OUTPUTS
# 1 - sample of size M
# INPUTS
# 'I' 2D histogram (or image)
# 'x' values on x coordinate
# 'y' values on y coordinate
# 'M' number of samples
function histogram2D_sampler(I, x, y, M)
    # dimensions of matrix
    r = size(I, 1);
    c = size(I, 2);
    # cartesian product of indices
    indices = collect(Iterators.product(1:r, 1:c));
    # vector of weights
    w = I[:];
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
