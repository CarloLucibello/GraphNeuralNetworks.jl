using SparseArrays
using GraphNeuralNetworks
using BenchmarkTools
import Random: seed!
using LinearAlgebra

n = 1024
seed!(0)
A = sprand(n, n, 0.01)
b = rand(1, n)
B = rand(100, n)

g = GNNGraph(A,
             ndata = (; b = b, B = B),
             edata = (; A = reshape(A.nzval, 1, :)),
             graph_type = :coo)

function spmv(g)
    propagate((xi, xj, e) -> e .* xj,  # same as e_mul_xj
              g, +; xj = g.ndata.b, e = g.edata.A)
end

function spmm1(g)
    propagate((xi, xj, e) -> e .* xj,  # same as e_mul_xj
              g, +; xj = g.ndata.B, e = g.edata.A)
end
function spmm2(g)
    propagate(e_mul_xj,
              g, +; xj = g.ndata.B, e = vec(g.edata.A))
end

# @assert isequal(spmv(g),  b * A)  # true
# @btime spmv(g)  # ~5 ms
# @btime b * A  # ~32 us

@assert isequal(spmm1(g), B * A)  # true
@assert isequal(spmm2(g), B * A)  # true
@btime spmm1(g)  # ~9 ms
@btime spmm2(g)  # ~9 ms
@btime B * A  # ~400 us

function spmm_copyxj_fused(g)
    propagate(copy_xj,
              g, +; xj = g.ndata.B)
end

function spmm_copyxj_unfused(g)
    propagate((xi, xj, e) -> xj,
              g, +; xj = g.ndata.B)
end

Adj = map(x -> x > 0 ? 1 : 0, A)
@assert spmm_copyxj_unfused(g) ≈ B * Adj
@assert spmm_copyxj_fused(g) ≈ B * Adj # bug fixed in #107

@btime spmm_copyxj_fused(g)  # 268.614 μs (22 allocations: 1.13 MiB)
@btime spmm_copyxj_unfused(g)  # 4.263 ms (52855 allocations: 12.23 MiB)
@btime B * Adj  # 196.135 μs (2 allocations: 800.05 KiB)

println()
