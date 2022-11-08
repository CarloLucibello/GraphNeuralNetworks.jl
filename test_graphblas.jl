using GraphNeuralNetworks
using SuiteSparseGraphBLAS
using LinearAlgebra, SparseArrays

g = rand_graph(10, 20, graph_type=:graphblas)
x = rand(2, 10)
m = GCNConv(2 => 3)
A = adjacency_matrix(g)
@assert A isa GBMatrix
@assert A + I isa GBMatrix
@assert Float32.(A) isa GBMatrix

m(g, x)