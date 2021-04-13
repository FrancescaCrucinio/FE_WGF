#= Kernel density estimatior for CT reconstructions
OUTPUTS
1 - KDE evaluated at KDEeval
INPUTS
'piSample' sample from π (Nx2 matrix)
'KDEeval' evaluation points (2 column matrix)
=#
function ct_kde(piSample, KDEeval)
    N = size(piSample, 1);
    # Silverman's plug in bandwidth
    bw1 = 1.06*Statistics.std(piSample[:, 1])*N^(-1/5);
    bw2 = 1.06*Statistics.std(piSample[:, 2])*N^(-1/5);

    KDEdensity = zeros(1, size(KDEeval, 1));
    for i=size(KDEeval, 1)
        KDEdensity[i] = mean(pdf(MvNormal(KDEeval[i, :], diagm([bw1^2; bw2^2])), piSample'))/(bw1*bw2);
    end

    return KDEdensity;
end
