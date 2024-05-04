module GNNlibCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GNNlib
using GNNlib.GNNGraphs
using GNNlib.GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 
import GNNlib: propagate

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

include("GNNGraphs/query.jl")
include("GNNGraphs/transform.jl")
include("GNNGraphs/utils.jl")
include("msgpass.jl")

end #module
