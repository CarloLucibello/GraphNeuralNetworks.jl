module GraphNeuralNetworks

using NNlib: similar
using LinearAlgebra: similar, fill!
using Statistics: mean
using LinearAlgebra
using SparseArrays
import KrylovKit
using CUDA
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor
using MacroTools: @forward
using NNlib, NNlibCUDA
using ChainRulesCore
import LightGraphs
using LightGraphs: AbstractGraph, outneighbors, inneighbors, is_directed, ne, nv, 
                  adjacency_matrix, degree

export
    # gnngraph
    GNNGraph,
    edge_index,
    node_feature, edge_feature, global_feature,
    adjacency_list, normalized_laplacian, scaled_laplacian,
    add_self_loops,

    # from LightGraphs
    adjacency_matrix, 

    # layers/msgpass

    # layers/basic
    GNNLayer,
    GNNChain,

    # layers/conv
    GCNConv,
    ChebConv,
    GraphConv,
    GATConv,
    GatedGraphConv,
    EdgeConv,
    GINConv,

    # layers/pool
    GlobalPool,
    LocalPool,
    TopKPool,
    topk_index



    
include("gnngraph.jl")
include("graph_conversions.jl")
include("utils.jl")

include("layers/msgpass.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/pool.jl")


end
