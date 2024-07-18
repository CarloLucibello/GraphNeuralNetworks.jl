module GraphNeuralNetworksCUDAExt

using CUDA
using Random, Statistics, LinearAlgebra
using GraphNeuralNetworks
using GraphNeuralNetworks.GNNGraphs
using GraphNeuralNetworks.GNNGraphs: COO_T, ADJMAT_T, SPARSE_T 
import GraphNeuralNetworks: propagate

const CUMAT_T = Union{CUDA.AnyCuMatrix, CUDA.CUSPARSE.CuSparseMatrix}

include("msgpass.jl")

end #module
