function mixture_CValpha(Nparticles, dt, Niter, alpha, x0, hSample, M)

    # data for gaussian mixture example
    f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
    h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
            pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
    g(x, y) = pdf.(Normal(x, 0.045), y);

    # remove non finite elements for entropy computation
    function remove_non_finite(x)
           return isfinite(x) ? x : 0
    end
    # values at which evaluate KDE
    KDEx = range(0, stop = 1, length = 1000);
    # reference values for KL divergence
    refY = range(0, stop = 1, length = 1000);
    # number of sub-groups
    L = size(hSample, 1);
    E = zeros(1, L);
    for l=1:L
        # get reduced sample
        hSampleL = hSample[1:end .!= l, :];
        hSampleL = hSampleL[:];
        # WGF
        x = wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha, x0, hSample, M, 0.5);
        # KDE
        RKDE = rks.kde(x[Niter, :], var"eval.points" = KDEx);
        KDEyWGF = abs.(rcopy(RKDE[3]));
        ent = -mean(remove_non_finite.(KDEyWGF .* log.(KDEyWGF)));
        # kl
        trueH = h.(refY);
        # approximated value
        delta = refY[2] - refY[1];
        hatH = zeros(1, length(refY));
        # convolution with approximated f
        # this gives the approximated value
        for i=1:length(refY)
            hatH[i] = delta*sum(g.(KDEx, refY[i]).*KDEyWGF);
        end
        kl = kl_divergence(trueH, hatH);
        E[l] = kl-alpha*ent;
        println("$l")
    end

    return mean(E);
end
