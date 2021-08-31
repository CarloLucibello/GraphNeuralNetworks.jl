using GraphNeuralNetworks
using GraphNeuralNetworks: sort_edge_index
using Flux
using CUDA
using Flux: gpu, @functor
using Flux: @functor
using LinearAlgebra, Statistics, Random
using NNlib
using LightGraphs
using Zygote
using Test
CUDA.allowscalar(false)

include("cuda/test_utils.jl")

tests = [
    "featured_graph",
    "layers/msgpass",
    "layers/conv",
    "layers/pool",
    "layers/misc",
]

!CUDA.functional() && @warn("CUDA unavailable, not testing GPU support")

# Testing all graph types. :sparse is a bit broken at the moment
@testset "GraphNeuralNetworks: graph format $graph_type" for graph_type in (:coo, :sparse, :dense)
    global GRAPH_T = graph_type
    for t in tests
        include("$t.jl")

        if CUDA.functional() && GRAPH_T != :sparse && isfile("cuda/$t.jl")
            include("cuda/$t.jl")
        end
    end
end
