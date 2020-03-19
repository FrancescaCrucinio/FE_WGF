function Ysamples_gamma(M)

    h(y) = 3/(y.^2 .+ 2).^(5/2);
    Yval = range(-4, 4, length = 2000);
    Hval = h.(Yval);
    cdfH = cumsum(Hval)./sum(Hval);
    FUN(u) = findfirst(u .< cdfH);
    indices = FUN.(rand(M, 1));
    y = Yval[indices];

    return y;
end

function wgf_gamma(dt, lambda, M, N)

    Niter = trunc(Int, 1/dt);
    x = zeros(Niter, N);
    drift = zeros(Niter-1, N);
    y = Ysamples_gamma(M);
    x[1, :] = 4*rand(1, N);
    for n=1:(Niter-1)
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sqrt.(x[n, :])), y[j]));
        end

        for i=1:N
            gradient = pdf.(Normal.(0, sqrt(x[n, i])), y) .* (y.^2 .- x[n, i])/(2*x[n, i]^2);
            drift[n, i] = mean(gradient./hN);
        end
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, drift
end
