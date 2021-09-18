module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra
using SparseArrays
import KrylovKit
using Base: tail
using CUDA
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor, batch
using MacroTools: @forward
import LearnBase
using LearnBase: getobs
using NNlib, NNlibCUDA
using ChainRulesCore
import LightGraphs
using LightGraphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree

export
    # gnngraph
    GNNGraph,
    edge_index,
    node_features, edge_features, graph_features,
    adjacency_list, normalized_laplacian, scaled_laplacian,
    add_self_loops, remove_self_loops,
    getgraph,

    # from LightGraphs
    adjacency_matrix, 
    # from SparseArrays
    sprand, sparse, 

    # msgpass
    update_node, update_edge, compute_message, propagate,

    # layers/basic
    GNNLayer,
    GNNChain,

    # layers/conv
    ChebConv,
    EdgeConv,
    GATConv,
    GatedGraphConv,
    GCNConv,
    GINConv,
    GraphConv,
    NNConv,

    # layers/pool
    GlobalPool,
    TopKPool,
    topk_index


    
include("gnngraph.jl")
include("graph_conversions.jl")
include("utils.jl")
include("msgpass.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/pool.jl")

end
