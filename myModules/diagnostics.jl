module diagnostics

using Statistics;
using StatsBase;

export diagnosticsH
export diagnosticsF

#=
Diagnostics for approximations of h
OUTPUTS
1 - Mean Integrated Square Error
2 - Kullback Leibler divergence
INPUTS
'h' true h (function handle)
'g' mixing kernel (function handle)
'KDEx' points in the domain of f at which the approximated f
and the true f are compared
'KDEy' approximated f
'refY' points in the domain of h at which the approximated h
and the true h are compared
=#
function diagnosticsH(h, g, KDEx, KDEy, refY)
    # distance between reference points
    delta = refY[2] - refY[1];
    # exact value
    trueH = h.(refY);
    # logarithm of true h for KL divergence
    trueHlog = log.(trueH);
    trueHlog[isnan.(trueHlog)] .= 0;
    hatH = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatH[i] = delta*sum(g.(KDEx, refY[i]).*KDEy);
    end
    # mise
    mise = var(trueH .- hatH, corrected = false);
    # compute log of hatH for Kl divergence
    hatHlog = log.(hatH);
    hatHlog[isnan.(hatHlog)] .= 0;
    div = sum(trueH.*(trueHlog .- hatHlog));

    return mise, div
end
#=
Diagnostics for approximations of f
OUTPUTS
1 - mean
2 - variance
3 - mean integrated squared error
INPUTS
'f' true f (function handle)
'x' sample points in the domain of f
'y' estimated value of f at sample points
=#
function diagnosticsF(f, x, y)
    #  mean
    m =  Statistics.mean(x, weights(y));
    # variance
    v = var(x, weights(y), corrected = false);
    # exact f
    trueF = f.(x);
    # compute MISE for f
    mise= var(trueF .- y, corrected = false);

    return m, v, mise
end

end
