module GraphNeuralNetworks

using Statistics: mean
using LinearAlgebra, Random
using Base: tail
using MacroTools: @forward
using MLUtils
using NNlib
using NNlib: scatter, gather
using ChainRulesCore
using SparseArrays, Graphs # not needed but if removed Documenter will complain

include("GNNGraphs/GNNGraphs.jl")

using .GNNGraphs

using .GNNGraphs: COO_T, ADJMAT_T, SPARSE_T,
                  check_num_nodes, check_num_edges,
                  EType, NType # for heteroconvs

export
# utils
      reduce_nodes,
      reduce_edges,
      softmax_nodes,
      softmax_edges,
      broadcast_nodes,
      broadcast_edges,
      softmax_edge_neighbors,

# msgpass
      apply_edges,
      aggregate_neighbors,
      propagate,
      copy_xj,
      copy_xi,
      xi_dot_xj,
      xi_sub_xj,
      xj_sub_xi,
      e_mul_xj,
      w_mul_xj,

# layers/basic
      dot_encoder,

# layers/conv
    agnn_conv,
    cgc_conv,
    cheb_conv,
    edge_conv,
    egnn_conv,
    gat_conv,
    gatv2_conv,
    gated_graph_conv,
    gcn_conv,
    gin_conv,
    gmm_conv,
    graph_conv,
    megnet_conv,
    nn_conv,
    res_gated_graph_conv,
    sage_conv,
    sg_conv,
    transformer_conv,

# # layers/heteroconv
#       HeteroGraphConv,

# layers/temporalconv
      TGCN,
      A3TGCN,

# layers/pool
      GlobalPool,
      GlobalAttentionPool,
      Set2Set,
      TopKPool,
      topk_index,

# mldatasets
      mldataset2gnngraph

include("utils.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/heteroconv.jl")
include("layers/temporalconv.jl")
include("layers/pool.jl")
include("msgpass.jl")
include("mldatasets.jl")

end
