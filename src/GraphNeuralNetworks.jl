module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
using Base: tail
using CUDA
using Flux
using Flux: glorot_uniform, leakyrelu, GRUCell, @functor, batch
using MacroTools: @forward
using NNlib, NNlibCUDA
using NNlib: scatter, gather
using ChainRulesCore
using Reexport

using SparseArrays, Graphs # not needed but if removed Documenter will complain

include("GNNGraphs/GNNGraphs.jl")
@reexport using .GNNGraphs
using .GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
                  check_num_nodes, check_num_edges

export
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


include("utils.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/pool.jl")
include("msgpass.jl")
include("deprecations.jl")

end
