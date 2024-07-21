module GNNGraphsCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNGraphs
using GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

include("query.jl")
include("transform.jl")
include("utils.jl")

end #module
