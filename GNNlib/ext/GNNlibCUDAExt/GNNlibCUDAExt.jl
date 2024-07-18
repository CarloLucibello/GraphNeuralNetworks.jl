module GNNlibCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
import GNNlib: propagate

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

include("msgpass.jl")

end #module
