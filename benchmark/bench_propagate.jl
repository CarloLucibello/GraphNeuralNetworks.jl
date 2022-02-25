module BenchPropagate
using GraphNeuralNetworks
using Random, Statistics, SparseArrays
using BenchmarkTools

SUITE = BenchmarkGroup()

Random.seed!(17)
n = 1024
A = sprand(n, n, 0.01)
v = rand(1, n)
X = rand(100, n)
w = reshape(A.nzval, 1, :)
e = rand(1, length(A.nzval))

g = GNNGraph(A, ndata=(; v, X), edata=(; e, w), graph_type=:coo)

spmv(g) = propagate((xi, xj, e) -> e .* xj, g, +; xj=g.ndata.v, e=g.edata.w)
spmm1(g) = propagate((xi, xj, e) -> e .* xj , g, +; xj=g.ndata.X, e=g.edata.w)
spmm2(g) = propagate(e_mul_xj, g, +; xj=g.ndata.X, e=vec(g.edata.w))

@assert spmv(g) ≈ v * A
@assert spmm1(g) ≈ X * A
@assert spmm2(g) ≈ X * A

let sub = SUITE["e_mul_xj"] = BenchmarkGroup()
    sub["spmv0"] = @benchmarkable $v * $A
    sub["spmv"] = @benchmarkable spmv($g)
    sub["spmm0"] = @benchmarkable $X * $A
    sub["spmm1"] = @benchmarkable spmm1($g)
    sub["spmm2"] = @benchmarkable spmm2($g)
end

spmm_copyxj_fused(g) = propagate(copy_xj, g, +; xj=g.ndata.X)
spmm_copyxj_unfused(g) = propagate((xi, xj, e) -> xj, g, +; xj=g.ndata.X)


let sub = SUITE["copy_xj"] = BenchmarkGroup()
    Adj = map(x -> x > 0 ? 1 : 0, A)
    @assert spmm_copyxj_unfused(g) ≈ X * Adj
    @assert spmm_copyxj_fused(g) ≈ X * Adj

    sub["copyxj0"] = @benchmarkable $X * $Adj
    sub["copyxj_fused"] = @benchmarkable spmm_copyxj_fused($g)
    sub["copyxj_unfused"] = @benchmarkable spmm_copyxj_unfused($g)
end

# @btime spmm_copyxj_fused(g)  # 268.614 μs (22 allocations: 1.13 MiB)
# @btime spmm_copyxj_unfused(g)  # 4.263 ms (52855 allocations: 12.23 MiB)
# @btime B * Adj  # 196.135 μs (2 allocations: 800.05 KiB)

end
BenchPropagate.SUITE