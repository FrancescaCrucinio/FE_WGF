#= Sobolev norm for WGF reconstructions
OUTPUTS
1 - integrated square error
2 - integrated square error for gradient
INPUTS
'piSample' sample from π (Nxd matrix)
'Nbins' number of bins for each dimension
'W' weights for piSample
'epsilon' regularisation parameter for SMC-EMS
=#
function sobolev_norm_kde(piSample, Nbins, W, epsilon::Float64 = 1e-3)
    # dimension
    d = size(piSample, 1);
    # sample size
    N = size(piSample, 2);
    # mixture of Gaussians
    means = [0.3*ones(1, d); 0.7*ones(1, d)];
    variances = [0.07^2; 0.1^2];
    pi = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d))), (means[2, :], diagm(variances[2]*ones(d)))], [1/3, 2/3]);
    sigmaK = 0.15;
    mu = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2, :], diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);

    # grid
    Xbins = range(1/Nbins, stop = 1-1/Nbins, length = Nbins);
    dx = Xbins[2] - Xbins[1];
    iter = Iterators.product((Xbins for _ in 1:d)...);
    KDEeval = reduce(vcat, vec([collect(i) for i in iter])');

    # Silverman's plug in bandwidth
    bw = zeros(d);
    for i=1:d
        if(all(y->y==W[1], W))
            bw[i] = 1.06*Statistics.std(piSample[i, :])*N^(-1/5);
        else
            bw[i] = sqrt(epsilon^2 + optimal_bandwidthESS(piSample[i, :], W)^2);
        end
    end

    approx_density = zeros(size(KDEeval, 1));
    approx_gradient = zeros(size(KDEeval, 1), d);
    for i = 1:size(KDEeval, 1)
        densityKDE = W.*pdf(MvNormal(KDEeval[i, :], Diagonal(bw.^2)), piSample)/prod(bw);
        approx_density[i] = pdf(pi, KDEeval[i, :]) - sum(densityKDE);
        approx_gradient[i, :] = pdf(components(pi)[1], KDEeval[i, :])/3 * (KDEeval[i, :] .- means[1, 1])./variances[1]^2 .+
            2*pdf(components(pi)[2], KDEeval[i, :])/3 * (KDEeval[i, :] .- means[2, 1])./variances[2]^2 .-
            sum(densityKDE' .* (KDEeval[i, :] .- piSample)./bw.^2, dims = 2);
    end
    norm_density = dx^d * sum(approx_density.^2);
    norm_gradient = dx^d * sum(approx_gradient.^2);
    return norm_density + norm_gradient
end
