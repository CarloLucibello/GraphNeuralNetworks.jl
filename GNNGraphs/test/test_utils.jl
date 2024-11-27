using ChainRulesTestUtils, FiniteDifferences, Zygote, Adapt, CUDA
CUDA.allowscalar(false)

function ngradient(f, x...)
    fdm = central_fdm(5, 1)
    return FiniteDifferences.grad(fdm, f, x...)
end
