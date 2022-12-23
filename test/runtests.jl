using GraphNeuralNetworks
using GraphNeuralNetworks.GNNGraphs: sort_edge_index
using GraphNeuralNetworks.GNNGraphs: getn, getdata
using Functors
using Flux
using CUDA
using Flux: gpu, @functor
using LinearAlgebra, Statistics, Random
using NNlib
import MLUtils
import StatsBase
using SparseArrays
using Graphs
using Zygote
using Test
using MLDatasets
using InlineStrings  # not used but with the import we test #98 and #104

CUDA.allowscalar(false)

const ACUMatrix{T} = Union{CuMatrix{T}, CUDA.CUSPARSE.CuSparseMatrix{T}}

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets

include("test_utils.jl")

tests = [
    "GNNGraphs/datastore",
    "GNNGraphs/gnngraph",
    "GNNGraphs/convert",
    "GNNGraphs/transform",
    "GNNGraphs/operators",
    "GNNGraphs/generate",
    "GNNGraphs/query",
    "GNNGraphs/sampling",
    # # "GNNGraphs/gnnheterograph",
    "utils",
    "msgpass",
    "layers/basic",
    "layers/conv",
    "layers/pool",
    "mldatasets",
    "examples/node_classification_cora",
    "deprecations",
]

!CUDA.functional() && @warn("CUDA unavailable, not testing GPU support")

@testset "GraphNeuralNetworks: graph format $graph_type" for graph_type in (:coo, :dense, :sparse) 
    global GRAPH_T = graph_type
    global TEST_GPU = CUDA.functional() && (GRAPH_T != :sparse)

    for t in tests
        startswith(t, "examples") && GRAPH_T == :dense && continue     # not testing :dense since causes OutOfMememory on github's CI
        include("$t.jl")
    end
end
