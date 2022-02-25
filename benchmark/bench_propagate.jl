module BenchPropagate
using GraphNeuralNetworks
using Random, Statistics, SparseArrays
using BenchmarkTools

SUITE = BenchmarkGroup()

Random.seed!(17)
n = 1024
A = sprand(n, n, 0.01)
X = rand(100, n)
w = reshape(A.nzval, 1, :)
e = rand(1, length(A.nzval))

g = GNNGraph(A, ndata=(; X), edata=(; e, w), graph_type=:coo)

spmm1(g) = propagate((xi, xj, e) -> e .* xj , g, +; xj=g.ndata.X, e=g.edata.w)
spmm2(g) = propagate(e_mul_xj, g, +; xj=g.ndata.X, e=vec(g.edata.w))

@assert spmm1(g) ≈ X * A
@assert spmm2(g) ≈ X * A

let sub = SUITE["e_mul_xj"] = BenchmarkGroup()
    sub["_baseline"] = @benchmarkable $X * $A
    sub["unfused"] = @benchmarkable spmm1($g)
    sub["fused"] = @benchmarkable spmm2($g)
end

spmm_copyxj_fused(g) = propagate(copy_xj, g, +; xj=g.ndata.X)
spmm_copyxj_unfused(g) = propagate((xi, xj, e) -> xj, g, +; xj=g.ndata.X)


let sub = SUITE["copy_xj"] = BenchmarkGroup()
    Adj = map(x -> x > 0 ? 1 : 0, A)
    @assert spmm_copyxj_unfused(g) ≈ X * Adj
    @assert spmm_copyxj_fused(g) ≈ X * Adj

    sub["_baseline"] = @benchmarkable $X * $Adj
    sub["fused"] = @benchmarkable spmm_copyxj_fused($g)
    sub["unfused"] = @benchmarkable spmm_copyxj_unfused($g)
end

end
BenchPropagate.SUITE