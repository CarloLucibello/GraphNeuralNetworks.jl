using CUDA, cuDNN
using GNNGraphs
using GNNGraphs: getn, getdata
using Functors
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
using SimpleWeightedGraphs
using LuxDeviceUtils: gpu_device, cpu_device, get_device
using LuxDeviceUtils: LuxCUDADevice # remove after https://github.com/LuxDL/LuxDeviceUtils.jl/pull/58

CUDA.allowscalar(false)

const ACUMatrix{T} = Union{CuMatrix{T}, CUDA.CUSPARSE.CuSparseMatrix{T}}

ENV["DATADEPS_ALWAYS_ACCEPT"] = true # for MLDatasets

include("test_utils.jl")

tests = [
    "chainrules",
    "datastore",
    "gnngraph",
    "convert",
    "transform",
    "operators",
    "generate",
    "query",
    "sampling",
    "gnnheterograph",
    "temporalsnapshotsgnngraph",
    "ext/SimpleWeightedGraphs"
]

!CUDA.functional() && @warn("CUDA unavailable, not testing GPU support")

for graph_type in (:coo, :dense, :sparse)
    @info "Testing graph format :$graph_type"
    global GRAPH_T = graph_type
    global TEST_GPU = CUDA.functional() && (GRAPH_T != :sparse)
    # global GRAPH_T = :sparse
    # global TEST_GPU = false

    @testset "$t" for t in tests
        include("$t.jl")
    end
end
