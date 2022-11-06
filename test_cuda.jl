
### how to make it work with CUDA.jl v4.0 ####
# dev Flux
# In the Flux project folder: 
    # - change the CUDA compat bound
    # - comment out usages of CUDA.has_cudnn()
# dev NNLibCUDA
# In the NNlibCUDA project folder: 
    # - change the CUDA compat bound
    # - add CUDA#master
    # - convert all the using CUDA.CUDNN to using CUDNN
    # - add https://github.com/JuliaGPU/CUDA.jl:lib/cudnn
# add CUDA#master
# add https://github.com/JuliaGPU/CUDA.jl:lib/cudnn # CUDNN subpackage not registered yet

using GraphNeuralNetworks, CUDA, Flux
using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays
using GraphNeuralNetworks.GNNGraphs: binarize
CUDA.allowscalar(false)

g_cpu = rand_graph(10, 10, graph_type = :sparse)
g = g_cpu |> gpu

a = adjacency_matrix(g, Float32)
# maximum(a)
# minimum(a)
# extrema(a)
# sum(a)

x = rand(2, 10) |> gpu
z = rand(10, 2) |> gpu

@assert x * z isa CuMatrix
@assert a .+ 1 isa CuMatrix
@assert tanh.(a) isa CuSparseMatrix
@assert a + a isa CuSparseMatrix
@assert mul!(deepcopy(z), a, z, 0, 1) isa CuArray
@assert mul!(deepcopy(x), x, a, 0, 1) isa CuArray
# @assert mm!('N', 'N', 0, a, z, 1, deepcopy(z), 'O') isa CuArray

@assert x * a isa CuMatrix
@assert a * z isa CuMatrix
# a * a
f(x) = x > 0
@assert f.(a) isa CuSparseMatrixCSC{Bool}
# map(f, a)
@assert binarize.(a) isa CuSparseMatrix
# show(a')
# CUDA.ones(10) .* a
# a .* CUDA.ones(10)


b = CuSparseMatrixCSR(a)
@assert x * z isa CuMatrix
@assert b .+ 1 isa CuMatrix
@assert tanh.(b) isa CuSparseMatrix
@assert b + b isa CuSparseMatrix
@assert x * b isa CuMatrix
@assert b * z isa CuMatrix
f(x) = x > 0
#BUG # @assert f.(b) isa CuSparseMatrixCSC{Bool}
# map(f, b)

c = CuSparseMatrixCOO(a)
@assert x * z isa CuMatrix
# BUG @assert c .+ 1 isa CuMatrix
# BUG @assert tanh.(c) isa CuSparseMatrix
# BUG @assert c + c isa CuSparseMatrix
@assert x * c isa CuMatrix
@assert c * z isa CuMatrix
f(x) = x > 0
# map(f, c)
# BUG @assert f.(c) isa CuSparseMatrixCSC{Bool}


# b * b
m = GCNConv(2 => 2) |> gpu
y = m(g, x)

g2 = rand_graph(10, 10, graph_type=:coo) |> gpu
adjacency_matrix(g2)


a