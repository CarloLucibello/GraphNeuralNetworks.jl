using Test
using Lux
using GNNLux
using Random, Statistics


tests = [
    # "utils",
    # "msgpass",
    # "layers/basic",
    "layers/conv",
    # "layers/heteroconv",
    # "layers/temporalconv",
    # "layers/pool",
    # "examples/node_classification_cora",
]

@testset "$t" for t in tests
    include("$t.jl")
end
