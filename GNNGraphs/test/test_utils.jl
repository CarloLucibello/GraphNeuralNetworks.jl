using ChainRulesTestUtils, FiniteDifferences, Zygote, Adapt, CUDA
CUDA.allowscalar(false)

# Using this until https://github.com/JuliaDiff/FiniteDifferences.jl/issues/188 is fixed
function FiniteDifferences.to_vec(x::Integer)
    Integer_from_vec(v) = x
    return Int[x], Integer_from_vec
end

function ngradient(f, x...)
    fdm = central_fdm(5, 1)
    return FiniteDifferences.grad(fdm, f, x...)
end
