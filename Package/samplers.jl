module samplers

using Statistics;
using StatsBase;
using IterTools;

struct histogram_object
       weights
       edges
   end

export histogram_sampler
export histogram_object
export image2histogram

# Build histogram from image
# OUTPUTS
# 1 - histogram object as given by StatsBase.fit(Histogram, data)
# INPUTS
# 'image' image or discretised function
# 'support' matrix containing the bounds of the support of the function (or image)
# each row gives the interval on one dimension
function image2histogram(image, support)
    # number of dimensions
    D = ndims(image);
    # build histogram object
    h = histogram_object(image, tuple([range(support[i, 1], stop = support[i, 2], length = size(image, i)+1) for i=1:D]...));
end
# Sample from histogram
# OUTPUTS
# 1 - sample of size M
# INPUTS
# 'h' histogram object obtained from StatsBase.fit(Histogram, data)
# 'M' number of samples
function histogram_sampler(h, M)
    # dimensions of histogram
    D = ndims(h.weights);
    # centres of bins
    bin_centres = reduce(vcat,[[h.edges[i][1:(end-1)] .+ (h.edges[i][2] - h.edges[i][1])/2] for i=1:D]);
    # iterators for bin centres
    bin_centres_iter = [1:size(h.weights, i) for i in 1:D];
    # cartesian product of indices
    indices = [collect(x) for x in Iterators.product(bin_centres_iter...)];
    # vector of weights
    w = h.weights[:];
    # walker sampler
    samples_indices = reduce(hcat, indices[walker_sampler(w, M)]);
    # odd entries are rows and even entries are columns
    samples = reduce(hcat, [bin_centres[i][samples_indices[i, :]]  for i in 1:D]);
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
