module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
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
using NNlib: scatter, gather
using ChainRulesCore
import Graphs
using Graphs: AbstractGraph, outneighbors, inneighbors, adjacency_matrix, degree

export
    # gnngraph
    GNNGraph,
    edge_index,
    node_features, edge_features, graph_features,
    adjacency_list, normalized_laplacian, scaled_laplacian,
    add_self_loops, remove_self_loops,
    getgraph,

    # from Graphs
    adjacency_matrix, 
    # from SparseArrays
    sprand, sparse, blockdiag,

    # utils
    reduce_nodes, reduce_edges, 
    softmax_nodes, softmax_edges,
    broadcast_nodes, broadcast_edges,
    softmax_edge_neighbors,
    
    # msgpass
    apply_edges, propagate,
    copyxj,

    # layers/basic
    GNNLayer,
    GNNChain,
    WithGraph,

    # layers/conv
    CGConv,
    ChebConv,
    EdgeConv,
    GATConv,
    GatedGraphConv,
    GCNConv,
    GINConv,
    GraphConv,
    NNConv,
    ResGatedGraphConv,
    SAGEConv,
    
    # layers/pool
    GlobalPool,
    GlobalAttentionPool,
    TopKPool,
    topk_index


include("gnngraph.jl")
include("graph_conversions.jl")
include("utils.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/pool.jl")
include("msgpass.jl")
include("deprecations.jl")

end
